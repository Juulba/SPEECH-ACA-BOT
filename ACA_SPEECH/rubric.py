import pandas as pd

class Rubric:
    def __init__(self, excel_filepath):
        self.df = self.load_excel_file(excel_filepath)

    def load_excel_file(self, filepath):
        df = pd.read_excel(filepath)
        df = df[df["Task"].notna()]
        df = df.reset_index(drop=True)

        df["Context"] = pd.Series([None] * len(df), dtype="object")
        df["Answer"] = pd.Series([None] * len(df), dtype="object")

        return df

    def get_next_question(self):
        nan_rows = self.df[self.df["Answer"].isna()]

        if not nan_rows.empty:
            return nan_rows.iloc[0]

        return "DONE"

    def get_topics_for_question(self, idx):
        if "Topic" not in self.df.columns:
            return "No explicit subtopics provided."

        topics_raw = self.df.loc[idx, "Topic"]

        if not isinstance(topics_raw, str) or not topics_raw.strip():
            return "No explicit subtopics provided."

        topics = [t.strip() for t in topics_raw.split(";") if t.strip()]
        return "\n".join(f"- {topic}" for topic in topics)