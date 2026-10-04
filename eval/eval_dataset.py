"""
Evaluation dataset contains the golden dataset of user queries and reference answers used to benchmark performance
"""
#reference (Ground Truth)

EVAL_DATASET_SEED = [
    # --- PUBMED ABSTRACTS from HuggingFace PubMedQA dataset --- 
    {
        'user_input': "Is smoking associated with increased HbA1c values and microalbuminuria in patients with diabetes -- data from the National Diabetes Register in Sweden?",
        'reference': "Smoking in patients with diabetes was widespread, especially in young female type 1, and in middle-aged type 1 and type 2 diabetes patients, and should be the target for smoking cessation campaigns. Smoking was associated with both poor glycaemic control and microalbuminuria, independently of other study characteristics." 
    },
    {
        'user_input': 'Does hyperglycaemia increase dipeptidyl peptidase IV activity in diabetes mellitus?',
        'reference': "Chronic hyperglycaemia induces a significant increase in DPP-IV activity in type 1 and type 2 diabetes. This phenomenon could contribute to the reduction in circulating active glucagon-like peptide-1 and to the consequent postprandial hyperglycaemia in type 2 diabetic patients with poor metabolic control."
    },
    {
        'user_input': 'Is the SPINK1 N34S mutation associated with Type 2 diabetes mellitus in a population of the USA?',
        'reference': "The SPINK1 N34S mutation appears not to predispose Hispanic or non-Hispanic white people from the USA to the development of Type 2 diabetes mellitus."
    },
    {
        'user_input': 'Does acute hyperglycaemia disturb cardiac repolarization in Type 1 diabetes?',
        'reference': "Acute hyperglycaemia alters myocardial ventricular repolarization in patients with Type 1 diabetes and in healthy volunteers and might consequently be an additional risk factor for cardiovascular events." 
    },
    {
        'user_input': 'Is hyperglycemia-induced platelet activation in type 2 diabetes resistant to aspirin but not to a nitric oxide-donating agent?',
        'reference': 'Acute hyperglycemia-induced enhancement of platelet activation is resistant to aspirin; a NO-donating agent suppresses it. Therapeutic approaches aiming at a wider platelet inhibitory action than that exerted by aspirin may prove useful in patients with type 2 diabetes.'
    },

    
# --- GENERATED GUIDELINE QA PAIRS ---


    {
        "user_input": "Under what circumstances should a clinician advise a patient with type 2 diabetes to stop pursuing their agreed-upon HbA1c target?",
        "reference": "Clinicians should encourage patients to reach and maintain their HbA1c target unless the efforts to achieve it or any resulting adverse effects, such as hypoglycaemia, impair the patient's quality of life."
    },
    {
        "user_input": "In an adult with type 2 diabetes, what are the treatment options if a DPP-4 inhibitor is contraindicated, not tolerated, or ineffective?",   
        "reference": "If a DPP-4 inhibitor is contraindicated, not tolerated, or not effective, consider pioglitazone or an insulin-based treatment."
    },
    {
        "user_input": "What specific steps and clinical risk factors should be assessed when examining the feet of a patient with diabetes?",
        "reference": "When examining the feet of a person with diabetes, you must remove their shoes, socks, bandages, and dressings. Both feet should be examined for the following risk factors: neuropathy (using a 10 g monofilament), limb ischaemia, ulceration, callus, infection and/or inflammation, deformity, gangrene, and Charcot arthropathy."
    },
    {
        "user_input": "Should GLP-1 receptor agonists be continued in patients with atherosclerotic cardiovascular disease or early onset type 2 diabetes if they are not achieving their individualized glycemic targets?",
        "reference": "Yes, for people with atherosclerotic cardiovascular disease or early onset type 2 diabetes, continuing GLP-1 receptor agonists can provide benefits in preventing cardiovascular events even if they do not help the patient reach their individualized glycemic targets."
    },
    {
        "user_input": "What is the clinical recommendation for the use of finerenone in adults with stage 3 or 4 chronic kidney disease and type 2 diabetes?",      
        "reference": "Finerenone is recommended as an option as an add-on to optimised standard care for some adults with stage 3 and 4 CKD (with ACR 3 mg/mmol or more) associated with type 2 diabetes, as detailed in NICE technology appraisal guidance TA877 (2023)."
    }
]
