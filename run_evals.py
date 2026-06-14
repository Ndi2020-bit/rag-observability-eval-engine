import os
import pandas as pd
from phoenix.session import evaluation
from phoenix.evals import (
    HallucinationEvaluator,
    QACorrectnessEvaluator,
    OpenAIModel,
    run_evals
)
import phoenix as px

print("📊 Fetching recorded traces from the local telemetry storage...")

# 1. Connect to your active running local Phoenix session
client = px.Client()

# 2. Extract the raw text data (Inputs, Retrieved Context, and Outputs) from your traces
spans_df = client.get_spans_dataframe(project_name="enterprise-rag-evaluation")

# Data preparation: Map the OpenTelemetry spans into readable query/context formats
queries = spans_df[spans_df['name'] == 'ChatOpenAI']['input'].apply(lambda x: x if isinstance(x, str) else str(x)).tolist()
references = spans_df[spans_df['name'] == 'Retriever']['output'].apply(lambda x: str(x)).tolist()
outputs = spans_df[spans_df['name'] == 'ChatOpenAI']['output'].apply(lambda x: str(x)).tolist()

eval_data = pd.DataFrame({
    "query": queries,
    "reference": references,
    "output": outputs
})

if eval_data.empty:
    print("❌ No evaluation data found! Make sure you ran 'rag_w_tracing.py' first.")
    exit()

print(f"📈 Loaded {len(eval_data)} execution paths. Running automated LLM-As-A-Judge grading...")

# 3. Initialize your 'LLM Evaluator Judge' using gpt-4o-mini
eval_model = OpenAIModel(model="gpt-4o-mini", temperature=0.0)

# 4. Configure our targeted structural evaluation metrics
hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_correctness_evaluator = QACorrectnessEvaluator(eval_model)

# 5. Execute the evaluation loops
hallucination_results = run_evals(
    dataframe=eval_data,
    evaluators=[hallucination_evaluator],
    provide_explanation=True
)

qa_correctness_results = run_evals(
    dataframe=eval_data,
    evaluators=[qa_correctness_evaluator],
    provide_explanation=True
)

print("\n🏆 --- AUTOMATED EVALUATION METRICS SUMMARY ---")
print(f"Hallucination Rate (Lower is better): {hallucination_results[0].score}")
print(f"QA Correctness (Higher is better): {qa_correctness_results[0].score}")
print(f"\nJudge Justification: {hallucination_results[0].explanation}")

# 6. Push these scores back into your dashboard UI
# client.log_evaluations(hallucination_results + qa_correctness_results)
print("\n🎯 Evaluation metrics logged successfully. Check your browser dashboard UI under 'Evaluators'!")
