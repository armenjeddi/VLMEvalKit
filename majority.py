import pandas as pd
from collections import Counter, defaultdict
import os
from openai import OpenAI

FAIL_MSG = 'Failed to obtain answer via API.'

def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""
    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""
    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""
    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""
    example_5 = """
Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""
    return [example_1, example_2, example_3, example_4, example_5]

def build_mathvista_gpt4_prompt(line):
    task_description = """
Please read the following example.
Then extract the answer from the model response and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}

def post_check(line, prefetch=False):
    res = None
    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']
    try:
        if line['question_type'] == 'multi_choice':
            ans = line['answer_option']
            choices = list_to_dict(eval(line['choices']))
            # Simplified can_infer - checks if response matches a choice
            res = str(response).strip().upper() if len(str(response).strip()) == 1 else None
            if res not in choices:
                res = None
            if prefetch:
                return res
        else:
            if line['answer_type'] == 'integer':
                res = int(response)
                ans = int(line['answer'])
            elif line['answer_type'] == 'float':
                res = float(response)
                ans = float(line['answer'])
            else:
                res = str(response)
                ans = str(ans)
    except (ValueError, KeyError, SyntaxError):
        pass

    if res == ans:
        return res if prefetch else True
    else:
        return False

def MathVista_acc(result_file):
    data = pd.read_excel(result_file) if isinstance(result_file, str) else result_file
    tot = defaultdict(lambda: 0)
    fetch = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    lt = len(data)
    skill_list = []
    for i in range(lt):
        item = data.iloc[i]
        cate = item['task']
        tot['Overall'] += 1
        try:
            skills = eval(item['skills']) if isinstance(item['skills'], str) else item['skills']
        except (SyntaxError, TypeError):
            skills = [item['skills']]
        for skill in skills:
            if skill not in skill_list:
                skill_list.append(skill)
            tot[skill] += 1
        tot[cate] += 1
        if item['log'] == 'Prefetch succeed':
            fetch['Overall'] += 1
            fetch[cate] += 1
            for skill in skills:
                fetch[skill] += 1
        if post_check(item, prefetch=False):
            hit['Overall'] += 1
            hit[cate] += 1
            for skill in skills:
                hit[skill] += 1

    res = defaultdict(list)
    for k in tot.keys():
        res['Task&Skill'].append(k)
        res['tot'].append(tot[k])
        res['prefetch'].append(fetch[k])
        res['hit'].append(hit[k])
        res['prefetch_rate'].append(fetch[k] / tot[k] * 100)
        res['acc'].append(hit[k] / tot[k] * 100)
    res = pd.DataFrame(res)
    return res

# New function to judge a single prediction with GPT
def judge_with_gpt(client, line, model="gpt-4"):
    """Judge a single prediction using GPT API"""
    # Check if we can extract directly
    if post_check(line, prefetch=True):
        res = post_check(line, prefetch=True)
        return dict(log='Prefetch succeed', res=str(res))
    
    prompt = build_mathvista_gpt4_prompt(line)
    
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=i * 0.5,
                max_tokens=50
            )
            res = response.choices[0].message.content.strip()
            
            if res and res != FAIL_MSG:
                return dict(log='Succeed', res=res)
        except Exception as e:
            print(f"API call failed (attempt {i+1}): {e}")
    
    return dict(log='All 5 retries failed.', res='')

def process_mathvista_results(excel_file, output_file, api_key, gpt_model="gpt-4"):
    """
    Process MathVista results with multiple predictions.
    
    Args:
        excel_file: Path to input Excel file with prediction_0, prediction_1, etc.
        output_file: Path to save output Excel file
        api_key: OpenAI API key
        gpt_model: GPT model to use for judging (default: gpt-4)
    """
    client = OpenAI(api_key=api_key)
    
    print("Loading data...")
    df = pd.read_excel(excel_file)
    
    pred_cols = [col for col in df.columns if col.startswith('prediction_')]
    pred_nums = sorted([int(col.split('_')[1]) for col in pred_cols])
    
    print(f"Found {len(pred_nums)} prediction columns: {pred_cols}")
    
    for num in pred_nums:
        pred_col = f'prediction_{num}'
        judged_col = f'judged_{num}'
        
        print(f"\nProcessing {pred_col}...")
        judged_results = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"  Processing row {idx}/{len(df)}")
            
            line = row.to_dict()
            line['prediction'] = row[pred_col]
            
            result = judge_with_gpt(client, line, model=gpt_model)
            judged_results.append(result['res'])
        
        df[judged_col] = judged_results
    
    voted_predictions = []
    
    judged_cols = [f'judged_{num}' for num in pred_nums]
    
    for idx, row in df.iterrows():
        predictions = [row[col] for col in judged_cols if pd.notna(row[col]) and row[col] != '']
        
        if predictions:
            counter = Counter(predictions)
            most_common = counter.most_common(1)[0][0]
            voted_predictions.append(most_common)
        else:
            voted_predictions.append('')
    
    df['voted_prediction'] = voted_predictions
    
    print(f"\nSaving results to {output_file}...")
    df.to_csv(output_file, index=False)
    
    eval_df = df.copy()
    eval_df['prediction'] = df['voted_prediction']
    eval_df['res'] = df['voted_prediction']
    
    acc_results = MathVista_acc(eval_df)
    
    print("ACCURACY RESULTS")
    print(acc_results.to_string(index=False))
    
    acc_output = output_file.replace('.csv', '_accuracy.csv')
    acc_results.to_csv(acc_output, index=False)
    print(f"\nAccuracy results saved to {acc_output}")
    
    return df, acc_results

if __name__ == "__main__":
    API_KEY=os.getenv("OPENAI_API_KEY")
    
    input_files = [
        '/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/dec31/MathVista_MINI_major5_temp07_kdvz05/Qwen/T20251231_G4587a723/Qwen_MathVista_MINI.xlsx',
        '/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/dec31/MathVista_MINI_major8_temp07/Qwen/T20251231_G4587a723/Qwen_MathVista_MINI.xlsx', 
        '/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/dec31/MathVista_MINI_major8_temp07_kdvz05/Qwen/T20251231_G4587a723/Qwen_MathVista_MINI.xlsx',

    ]
    
    output_files = [
        '/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/dec31/MathVista_MINI_major5_temp07_kdvz05/Qwen/T20251231_G4587a723/OpenAI_Qwen_MathVista_MINI.csv',
        '/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/dec31/MathVista_MINI_major8_temp07/Qwen/T20251231_G4587a723/OpenAI_Qwen_MathVista_MINI.csv', 
        '/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/dec31/MathVista_MINI_major8_temp07_kdvz05/Qwen/T20251231_G4587a723/OpenAI_Qwen_MathVista_MINI.csv',
    ]

    for INPUT_FILE, OUTPUT_FILE in zip(input_files, output_files):
        print(f"\nProcessing file: {INPUT_FILE}")
        df, acc_results = process_mathvista_results(
            excel_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            api_key=API_KEY,
            gpt_model="gpt-4o-mini"
        )
