import pandas as pd
from collections import Counter, defaultdict
import os
import copy
import string
import logging
import timeout_decorator
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import json
import numpy as np

# Try importing latex2sympy for MathVision
try:
    from latex2sympy2_extended import latex2sympy
except ImportError:
    try:
        from latex2sympy2 import latex2sympy
    except Exception:
        print("Warning: latex2sympy2 not found. MathVision evaluation will fail on equation checks.")

load_dotenv(Path(__file__).resolve().parent / ".env")

FAIL_MSG = 'Failed to obtain answer via API.'

# ==========================================
# SHARED HELPER FUNCTIONS (can_infer)
# ==========================================

def list_to_dict(lst):
    return {chr(65 + i): val for i, val in enumerate(lst)}

def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)

def can_infer_text(answer, choices):
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False

def can_infer_option(answer, choices):
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = copy.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                return False
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False

# ==========================================
# MATHVISION SPECIFIC FUNCTIONS
# ==========================================

@timeout_decorator.timeout(30)
def is_equal_mathvision(asw: str, gt_asw: str) -> bool:
    if not isinstance(asw, str) or not isinstance(gt_asw, str):
        return False
    asw = str(asw).lower().strip()
    gt_asw = str(gt_asw).lower().strip()
    if gt_asw == asw:
        return True
    try:
        a = eval(gt_asw)
        b = eval(asw)
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    try:
        a = latex2sympy(gt_asw)
        b = latex2sympy(asw)
        if abs(eval(str(a)) - eval(str(b))) < 1e-6:
            return True
        if abs(a - b) < 1e-6:
            return True
    except:
        pass
    return False

def get_mathvision_gpt4_ICE():
    return [
        """Hint: Please answer the question and provide the final answer at the end.\nQuestion: Which number is missing?\nModel response: The number missing in the sequence is 14.\nExtracted answer: 14""",
        """Hint: Please answer the question and provide the final answer at the end.\nQuestion: What is the fraction of females facing the camera?\nModel response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.\nExtracted answer: 0.6""",
        """Hint: Please answer the question and provide the final answer at the end.\nQuestion: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\nModel response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\nExtracted answer: 1.45""",
        """Hint: Please answer the question and provide the final answer at the end.\nQuestion: Between which two years does the line graph saw its maximum peak?\nModel response: The line graph saw its maximum peak between 2007 and 2008.\nExtracted answer: [2007, 2008]""",
        """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: What fraction of the shape is blue?\nChoices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\nModel response: The correct answer is (B) 8/11.\nExtracted answer: B"""
    ]

def build_mathvision_gpt4_prompt(line):
    task_description = """Please read the following example.\nThen extract the answer from the model response and type it at the end of the prompt.\n"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_mathvision_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def post_check_mathvision(line, prefetch=False):
    res = None
    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']
    
    try:
        choices_list = eval(line['choices']) if isinstance(line['choices'], str) else line['choices']
        if len(choices_list) > 0:
            ans = line['answer']
            choices = list_to_dict(choices_list)
            res = can_infer(response, choices)
            if prefetch:
                return res
        else:
            res = str(response)
            ans = str(ans)
    except (ValueError, SyntaxError):
        pass

    try:
        if is_equal_mathvision(res, ans):
            return res if prefetch else True
        else:
            return False
    except Exception as err:
        return False

def MathVision_acc(df):
    tot = defaultdict(lambda: 0)
    fetch = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    
    for _, item in df.iterrows():
        cate = item['category'] # MathVision uses 'category'
        tot['Overall'] += 1
        tot[cate] += 1
        
        # In this script, we don't track 'Prefetch succeed' explicitly in the dataframe 
        # unless we add that column, but for accuracy calculation:
        if post_check_mathvision(item, prefetch=False):
            hit['Overall'] += 1
            hit[cate] += 1

    res = defaultdict(list)
    for k in tot.keys():
        res['Subject'].append(k)
        res['tot'].append(tot[k])
        res['hit'].append(hit[k])
        res['acc'].append(hit[k] / tot[k] * 100)
    
    res = pd.DataFrame(res).sort_values('Subject', ignore_index=True)
    return res

# ==========================================
# MATHVISTA SPECIFIC FUNCTIONS
# ==========================================

def get_mathvista_gpt4_ICE():
    return [
        """Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.\nQuestion: Which number is missing?\nModel response: The number missing in the sequence is 14.\nExtracted answer: 14""",
        """Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.\nQuestion: What is the fraction of females facing the camera?\nModel response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.\nExtracted answer: 0.6""",
        """Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.\nQuestion: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\nModel response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\nExtracted answer: 1.45""",
        """Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\nQuestion: Between which two years does the line graph saw its maximum peak?\nModel response: The line graph saw its maximum peak between 2007 and 2008.\nExtracted answer: [2007, 2008]""",
        """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: What fraction of the shape is blue?\nChoices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\nModel response: The correct answer is (B) 8/11.\nExtracted answer: B"""
    ]

def build_mathvista_gpt4_prompt(line):
    task_description = """Please read the following example.\nThen extract the answer from the model response and type it at the end of the prompt.\n"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_mathvista_gpt4_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += question + '\n'
    prompt += 'Model respone: ' + prediction
    prompt += 'Extracted answer:'
    return prompt

def post_check_mathvista(line, prefetch=False):
    res = None
    ans = line['answer']
    response = line['prediction'] if prefetch else line['res']
    try:
        if line['question_type'] == 'multi_choice':
            ans = line['answer_option']
            choices = list_to_dict(eval(line['choices']))
            res = can_infer(response, choices)
            
            if not res:
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
    except (ValueError, KeyError, SyntaxError, TypeError):
        pass

    if res == ans:
        return res if prefetch else True
    else:
        return False

def MathVista_acc(df):
    tot = defaultdict(lambda: 0)
    hit = defaultdict(lambda: 0)
    skill_list = []
    
    for _, item in df.iterrows():
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
        
        if post_check_mathvista(item, prefetch=False):
            hit['Overall'] += 1
            hit[cate] += 1
            for skill in skills:
                hit[skill] += 1

    res = defaultdict(list)
    for k in tot.keys():
        res['Task&Skill'].append(k)
        res['tot'].append(tot[k])
        res['hit'].append(hit[k])
        res['acc'].append(hit[k] / tot[k] * 100)
    res = pd.DataFrame(res)
    return res


# ==========================================
# MATHVERSE SPECIFIC FUNCTIONS
# ==========================================

def get_mathverse_gpt4_extract_ICE():
    return [
        "1.\nModel response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'\nExtracted Answer: (-2, 1)\n",
        "2.\nModel response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.\",'\nExtracted Answer: D\n",
        "3.\nModel response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'\nExtracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)\n",
        "4.\nModel response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'\nExtracted Answer: null\n",
        "5.\nModel response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'\nExtracted answer: 22.3\n",
        "6.\nModel response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)\"'\nExtracted answer: f(x) = -x^2 - 2x + 1\n"
    ]

def get_mathverse_gpt4_score_ICE():
    return [
        "\n[Question]: Write the set of numbers represented on the number line in interval notation.\n[Standard Answer]: (-2,1]\n[Model_answer] : Extracted Answer: \\((-2, 1)\\)\nJudgement: 0\n",
        "\n[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}\n[Standard Answer]: C\n[Model_answer] : B:2\u221a{{3}}\nJudgement: 0\n",
        "\n[Question]: Find the domain and range of the function f using interval notation.\n[Standard Answer]: domain: [-4, 0) and range: (-3, 1]\n[Model_answer] : Range: \\((-4, 1]\\)\nJudgement: 0\n",
        "\n[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}\n[Standard Answer]: C\n[Model_answer] : null\nJudgement: 0\n"
    ]

def build_mathverse_gpt4_extract_prompt(line):
    task_description = """I am providing you a response from a model to a math problem, termed 'Model Response'. You should extract the answer from the response as 'Extracted Answer'. Directly output the extracted answer with no explanation.\n\n"""
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_mathverse_gpt4_extract_ICE()
    for example in examples:
        prompt += example + '\n'
    prompt += f"Model response: '{prediction}'\nExtracted Answer: "
    return prompt

def build_mathverse_gpt4_score_prompt(line):
    task_description = """Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judement is 1; if they are different, Judement is 0.\n\n"""
    question_for_eval = line.get('question_for_eval', line.get('question', '')) 
    extract = line['extract']
    answer = line['answer']
    
    prompt = task_description
    examples = get_mathverse_gpt4_score_ICE()
    for example in examples:
        prompt += example + '\n'
    
    prompt += f"""
    [Question]: {question_for_eval}
    [Standard Answer]: {answer}
    [Model_answer] : {extract}
    Judgement:"""
    return prompt

def post_check_mathverse(line, client=None):
    """
    Checks correctness. 
    1. Tries exact string match.
    2. If fails and client is provided, uses GPT-4 to judge (score).
    """
    ans = str(line['answer']).strip()
    # In the accuracy loop, 'prediction' contains the extracted answer
    response = str(line['prediction']).strip()

    if response == ans:
        return True
    
    if client:
        eval_line = {
            'question': line.get('question', ''),
            'question_for_eval': line.get('question', ''),
            'answer': ans,
            'extract': response
        }
        prompt = build_mathverse_gpt4_score_prompt(eval_line)
        
        for i in range(5):
            try:
                api_res = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=i * 0.5,
                    max_tokens=10
                )
                res_text = api_res.choices[0].message.content.strip()
                if res_text in ['0', '1']:
                    return int(res_text) == 1
            except Exception:
                pass
    
    return False

def MathVerse_acc(df, client):
    """
    Calculates accuracy for MathVerse.
    Requires parsing 'metadata' JSON for subfields/subjects and running LLM scoring.
    """
    print("Calculating MathVerse Accuracy (this involves LLM scoring for mismatches)...")
    
    try:
        if isinstance(df.iloc[0]['metadata'], str):
            df['metadata'] = df['metadata'].apply(lambda x: x.replace("'", '"') if isinstance(x, str) else x)
            df['metadata'] = df['metadata'].apply(json.loads)
    except Exception as e:
        print(f"Warning: Metadata parsing failed or already dict: {e}")

    meta_df = pd.json_normalize(df['metadata'])
    df = df.reset_index(drop=True)
    meta_df = meta_df.reset_index(drop=True)
    df = pd.concat([df, meta_df], axis=1)

    scores = []
    for idx, row in df.iterrows():
        if idx % 50 == 0:
            print(f"  Scoring row {idx}/{len(df)}")
        is_correct = post_check_mathverse(row, client=client)
        scores.append(1 if is_correct else 0)
    
    df['score'] = scores

    subset = list(set(df['problem_version']))
    if 'Overall' not in subset:
        subset.append('Overall')

    res = defaultdict(list)
    for p in subset:
        if p != 'Overall':
            sub = df[df['problem_version'] == p]
        else:
            sub = df.copy()
            
        res['split'].append(p)
        res['Overall'].append(np.mean(sub['score']) * 100)
        
        if 'subject' in df.columns:
            subjects = set(df['subject'].dropna())
            for k in subjects:
                sub_k = sub[sub['subject'] == k]
                acc = np.mean(sub_k['score']) * 100 if len(sub_k) > 0 else 0
                res[k].append(acc)
                
        if 'subfield' in df.columns:
            subfields = set(df['subfield'].dropna())
            for k in subfields:
                sub_k = sub[sub['subfield'] == k]
                acc = np.mean(sub_k['score']) * 100 if len(sub_k) > 0 else 0
                res[k].append(acc)

    return pd.DataFrame(res)



# ==========================================
# UNIFIED JUDGING & PROCESSING LOGIC
# ==========================================

def judge_with_gpt(client, line, model="gpt-4", dataset="MathVista"):
    """Judge a single prediction using GPT API, dispatching to correct dataset logic."""
    
    # Dispatcher
    if dataset == "MathVision":
        checker = post_check_mathvision
        prompter = build_mathvision_gpt4_prompt
    elif dataset == "MathVerse":
        # MathVerse specific: logic is split. 
        # Here we only do Extraction. Scoring happens in accuracy calc.
        checker = lambda x, prefetch: False # Disable prefetch for MathVerse extraction to force GPT prompt
        prompter = build_mathverse_gpt4_extract_prompt
    else:
        checker = post_check_mathvista
        prompter = build_mathvista_gpt4_prompt

    # 1. Prefetch / Heuristic Check
    # (Skip for MathVerse extraction to ensure clean formatting via LLM)
    if dataset != "MathVerse" and checker(line, prefetch=True):
        res = checker(line, prefetch=True)
        return dict(log='Prefetch succeed', res=str(res))
    
    # 2. GPT Extraction
    prompt = prompter(line)
    
    for i in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=i * 0.5,
                max_tokens=256
            )
            res = response.choices[0].message.content.strip()
            
            if dataset == "MathVerse":
                if "Extracted Answer:" in res:
                    res = res.split("Extracted Answer:")[-1].strip()

            if res and res != FAIL_MSG:
                return dict(log='Succeed', res=res)
        except Exception as e:
            print(f"API call failed (attempt {i+1}): {e}")
    
    return dict(log='All 5 retries failed.', res='')


def process_results(excel_file, output_file, api_key, gpt_model="gpt-4", dataset="MathVista"):
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
            if idx % 100 == 0:
                print(f"  Processing row {idx}/{len(df)}")
            
            line = row.to_dict()
            line['prediction'] = row[pred_col]
            
            result = judge_with_gpt(client, line, model=gpt_model, dataset=dataset)
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
    
    if dataset == "MathVision":
        acc_results = MathVision_acc(eval_df)
    elif dataset == "MathVerse":
        acc_results = MathVerse_acc(eval_df, client)
    else:
        acc_results = MathVista_acc(eval_df)
    
    print("ACCURACY RESULTS")
    print(acc_results.to_string(index=False))
    
    acc_output = output_file.replace('.csv', '_accuracy.csv')
    acc_results.to_csv(acc_output, index=False)
    print(f"\nAccuracy results saved to {acc_output}")
    
    return df, acc_results


if __name__ == "__main__":
    API_KEY = os.environ.get('OPENAI_API_KEY')
    DATASET_TYPE = "MathVista"

    input_files = [
       "/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/jan08/MathVista_MINI_major5_temp07/InternVL/T20260108_Gf5ccea51/InternVL_MathVista_MINI.xlsx",
       "/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/jan08/MathVista_MINI_major5_temp07_kdvz05/InternVL/T20260108_Gf5ccea51/InternVL_MathVista_MINI.xlsx"
    ]
    output_files = [
        "/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/jan08/MathVista_MINI_major5_temp07/InternVL/T20260108_Gf5ccea51/openai_InternVL_MathVista_MINI.csv",
        "/home/minhle/projects/aip-btaati/minhle/VLMEvalKit/results/jan08/MathVista_MINI_major5_temp07_kdvz05/InternVL/T20260108_Gf5ccea51/openai_InternVL_MathVista_MINI.csv"
    ]
    

    for INPUT_FILE, OUTPUT_FILE in zip(input_files, output_files):
        print(f"\nProcessing file: {INPUT_FILE}")
        df, acc_results = process_results(
            excel_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            api_key=API_KEY,
            gpt_model="gpt-4o-mini",
            dataset=DATASET_TYPE
        )