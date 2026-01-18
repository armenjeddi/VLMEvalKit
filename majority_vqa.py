import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import string
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from vlmeval.smp import *
from vlmeval.dataset.utils.yorn import YOrN_Extraction, YOrN_match_prompt
from vlmeval.dataset.utils.multiple_choice import extract_characters_regex, build_prompt, MMB_abbrs

load_dotenv(Path(__file__).resolve().parent / ".env")

FAIL_MSG = 'Failed to obtain answer via API.'


def MME_rating(df):
    stats = defaultdict(dict)
    
    for _, item in df.iterrows():
        category = item['category']
        image_path = item['image_path']

        is_correct = (str(item['answer']).lower() == str(item['extracted']).lower())
        score = 1 if is_correct else 0
        
        if image_path not in stats[category]:
            stats[category][image_path] = []
        stats[category][image_path].append(score)


    def acc(key, mode='normal'):
        res = stats[key]
        values = []
        for val in res.values():
            values.extend(val)
            if mode == 'normal':
                values.extend(val)
            elif mode == 'plus':
                # MME logic: Both questions for the image must be correct
                if len(val) > 1:
                    values.append(val[0] * val[1])
                else:
                    # Fallback if only 1 question exists for the image (rare in MME)
                    values.append(val[0])
        if len(values) == 0:
            return 0
        return np.mean(values) * 100

    scores = {}
    for k in stats:
        scores[k] = acc(k) + acc(k, 'plus')

    super_cates = dict(
        perception=[
            'OCR', 'artwork', 'celebrity', 'color', 'count', 'existence',
            'landmark', 'position', 'posters', 'scene'
        ],
        reasoning=['code_reasoning', 'commonsense_reasoning', 'numerical_calculation', 'text_translation']
    )

    ret = {}
    for sc, cate_list in super_cates.items():
        base = 0
        for c in cate_list:
            if c in scores:
                base += scores[c]
        ret[sc] = base
    ret.update(scores)
    ret = d2df(ret)
    return ret


def POPE_rating(data):
    def cal_f1_score(y_true, y_pred):
        tp = sum((y_true == 1) & (y_pred == 1))
        fp = sum((y_true == 0) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return f1_score, precision, recall

    # data = data_file
    data = data.assign(category=data['category'].str.split(',')).explode('category')
    data['index'] = range(len(data))
    res = dict(split=[], Overall=[], acc=[], precision=[], recall=[])
    y_true = np.array([1 if i == 'Yes' else 0 for i in data['answer']])
    y_pred = np.array([1 if i == 'Yes' else 0 for i in data['extracted']])
    f1_score, precision, recall = cal_f1_score(y_true, y_pred)
    res['split'].append('Overall')
    res['Overall'].append(f1_score * 100)
    res['acc'].append(np.mean(data['score']) * 100)
    res['precision'].append(precision * 100)
    res['recall'].append(recall * 100)

    if 'category' in data:
        cates = list(set(data['category']))
        cates = [c for c in cates if not pd.isna(c)]
        for c in cates:
            sub = data[data['category'] == c]
            y_true = np.array([1 if i == 'Yes' else 0 for i in sub['answer']])
            y_pred = np.array([1 if i == 'Yes' else 0 for i in sub['extracted']])
            f1_score, precision, recall = cal_f1_score(y_true, y_pred)
            res['split'].append(c)
            res['Overall'].append(f1_score * 100)
            res['acc'].append(np.mean(sub['score']) * 100)
            res['precision'].append(precision * 100)
            res['recall'].append(recall * 100)

    ret = pd.DataFrame(res)
    return ret

def MMStar_rating(df):
    
    results = {}
    results['Overall'] = df['score'].mean()
    groups_to_check = ['category', 'l2-category']
    
    for group in groups_to_check:
        if group not in df.columns:
            continue
            
        abilities = df[group].unique()
        
        for ab in abilities:
            ab_name = ab
            if 'MMB_abbrs' in globals() and ab in MMB_abbrs:
                ab_name = MMB_abbrs[ab]
            sub_score = df[df[group] == ab]['score'].mean()
            
            results[ab_name] = sub_score

    return pd.DataFrame([results])

# ==========================================
# UNIFIED JUDGING & PROCESSING LOGIC
# ==========================================

def judge_with_gpt(client, line, model="gpt-4", dataset="MME"):
    """
    Judge a single prediction. 
    1. Try yes/no extraction or multiple-choice extraction depending on dataset.
    2. If extraction fails, use GPT API with retries to get the answer.
    """

    prediction = line['prediction']

    if dataset in ["MME", "POPE"]:
        res = YOrN_Extraction(prediction)
        if res != 'Unknown':
            return dict(log='Rule-based succeed', res=res)
        
        prompt = YOrN_match_prompt(line)
        
        for i in range(5):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=i * 0.5,
                    max_tokens=10
                )
                gpt_res = response.choices[0].message.content.strip()
                
                final_res = YOrN_Extraction(gpt_res)
                
                if final_res != 'Unknown':
                    return dict(log='GPT succeed', res=final_res)
                    
            except Exception as e:
                print(f"API call failed (attempt {i+1}): {e}")
        
        return dict(log='All retries failed', res='Unknown')
    
    elif dataset in ["MMStar", "MMBench", "CV-Bench"]:
        res = extract_characters_regex(prediction, choices=['(A)', '(B)', '(C)', '(D)', '(E)', '(F)'])
        
        if res != '':
            return dict(log='Rule-based succeed', res=res)
                
        question = line['question']
        if dataset in ["MMBench", "MMStar"]:
            choices = ['A. ' + str(line['A']), 'B. ' + str(line['B']), 'C. ' + str(line['C']), 'D. ' + str(line['D'])]
        else: # CV-Bench
            choices = ['A. ' + str(line['A']), 'B. ' + str(line['B']), 'C. ' + str(line['C']), 
                       'D. ' + str(line['D']), 'E. ' + str(line['E']), 'F. ' + str(line['F'])]
        prediction = line['prediction']
        prompt = build_prompt(question, choices, prediction)

        for i in range(5):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=i * 0.5,
                    max_tokens=10
                )
                gpt_res = response.choices[0].message.content.strip()

                final_res = extract_characters_regex(gpt_res, choices=['(A)', '(B)', '(C)', '(D)', '(E)', '(F)'])

                if final_res != '':
                    return dict(log='GPT succeed', res=final_res)
            except Exception as e:
                print(f"API call failed (attempt {i+1}): {e}")
            
        return dict(log='All retries failed', res='Unknown')



def process_results(excel_file, output_file, api_key, gpt_model="gpt-4", dataset="MME"):
    client = OpenAI(api_key=api_key)
    
    print(f"Loading data from {excel_file} for dataset: {dataset}...")
    
    if str(excel_file).endswith('.csv'):
        df = pd.read_csv(excel_file)
    else:
        df = pd.read_excel(excel_file)
    
    pred_cols = [col for col in df.columns if col.startswith('prediction_')]
    pred_nums = sorted([int(col.split('_')[1]) for col in pred_cols])
    
    print(f"Found {len(pred_nums)} prediction columns: {pred_cols}")
    
    for num in pred_nums:
        pred_col = f'prediction_{num}'
        judged_col = f'judged_{num}'
        
        if judged_col in df.columns and not df[judged_col].isnull().all():
            print(f"Skipping {pred_col}, {judged_col} already exists.")
            continue

        print(f"\nProcessing {pred_col}...")
        judged_results = []
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                print(f"  Processing row {idx}/{len(df)}")
            
            line = row.to_dict()
            line['prediction'] = row[pred_col]
            
            result = judge_with_gpt(client, line, model=gpt_model, dataset=dataset)
            judged_results.append(result['res'])
        
        df[judged_col] = judged_results
    
    voted_predictions = []
    judged_cols = [f'judged_{num}' for num in pred_nums]
    
    print("\nPerforming Majority Voting...")
    for idx, row in df.iterrows():
        predictions = [row[col] for col in judged_cols if pd.notna(row[col])]
        
        if predictions:
            counter = Counter(predictions)
            most_common = counter.most_common(1)[0][0]
            voted_predictions.append(most_common)
        else:
            voted_predictions.append('Unknown')
    
    df['voted_prediction'] = voted_predictions
    
    
    
    df['extracted'] = df['voted_prediction']
    df['score'] = (df['answer'] == df['extracted'])

    print(f"\nSaving processed results to {output_file}...")
    df.to_csv(output_file, index=False)

    
    print("\nCalculating Accuracy")
    eval_df = df.copy()
    
    if dataset == "MME":
        acc_results = MME_rating(eval_df)
    elif dataset == "POPE":
        acc_results = POPE_rating(eval_df)
    elif dataset == "MMStar":
        acc_results = MMStar_rating(eval_df)
    
    print("ACCURACY RESULTS")
    print(acc_results.to_string(index=False))
    
    acc_output = output_file.replace('.csv', '_accuracy.csv')
    acc_results.to_csv(acc_output, index=False)
    print(f"\nAccuracy results saved to {acc_output}")
    
    return df, acc_results

if __name__ == "__main__":
    API_KEY = os.environ.get('OPENAI_API_KEY')

    input_files = [
       "/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/jan15/MMStar_kdvz075/Qwen/Qwen_MMStar.xlsx"]
    output_files = [
        "/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/jan15/MMStar_kdvz075/Qwen/processed_Qwen_MMStar.csv"

    ]

    
    for INPUT_FILE, OUTPUT_FILE in zip(input_files, output_files):
        if not os.path.exists(INPUT_FILE):
            print(f"File not found: {INPUT_FILE}")
            continue
        
        if "MME" in INPUT_FILE:
            DATASET_TYPE = "MME"
        elif "POPE" in INPUT_FILE:
            DATASET_TYPE = "POPE"
        elif "MMStar" in INPUT_FILE:
            DATASET_TYPE = "MMStar"
        elif "MMBench" in INPUT_FILE:
            DATASET_TYPE = "MMBench"
        elif "CV-Bench" in INPUT_FILE:
            DATASET_TYPE = "CV-Bench"
             
        print(f"\nProcessing file: {INPUT_FILE}")
        df, acc_results = process_results(
            excel_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            api_key=API_KEY,
            gpt_model="gpt-4o-mini",
            dataset=DATASET_TYPE
        )
