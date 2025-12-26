import pandas as pd
from rag_pipeline import RAGPipeline
from typing import List, Dict

class Evaluator:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.results = []

    def evaluate_questions(self, questions: List[Dict[str, str]], top_k: int = 3):
        """
        questions: list of dicts like [{"question": "...", "answer": "..."}]
        top_k: number of contexts to retrieve
        """
        for q in questions:
            res = self.rag_pipeline.query(q['question'], top_k)
            correct_answer = q['answer'].strip().lower()
            generated_answer = res['answer'].strip().lower()

            # Simple evaluation
            if correct_answer in generated_answer:
                status = "Correct"
            elif any(word in generated_answer for word in correct_answer.split()):
                status = "Partially Correct"
            else:
                status = "Incorrect"

            # Check if correct answer exists in retrieved contexts
            context_hit = any(correct_answer in c.lower() for c in res['contexts'])
            self.results.append({
                "Question": q['question'],
                "Expected Answer": q['answer'],
                "Generated Answer": res['answer'],
                "Context Hit": "Yes" if context_hit else "No",
                "Answer Status": status,
                "Latency (ms)": res['latency_ms']
            })

    def get_results_df(self):
        """Return a Pandas DataFrame of evaluation results"""
        return pd.DataFrame(self.results)

    def summary(self):
        df = self.get_results_df()
        total = len(df)
        correct = len(df[df['Answer Status'] == "Correct"])
        partial = len(df[df['Answer Status'] == "Partially Correct"])
        incorrect = len(df[df['Answer Status'] == "Incorrect"])
        avg_latency = df['Latency (ms)'].mean()

        summary_text = f"""
        Evaluation Summary:
        -------------------
        Total Questions: {total}
        Correct: {correct} ({correct/total*100:.1f}%)
        Partially Correct: {partial} ({partial/total*100:.1f}%)
        Incorrect: {incorrect} ({incorrect/total*100:.1f}%)
        Average Latency: {avg_latency:.0f} ms
        """
        return summary_text
