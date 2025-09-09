import pandas as pd
import re
from rapidfuzz import fuzz

class ManualClassification:
    def __init__(self, threshold: int = 85):
        self.threshold = threshold  # similarity threshold for fuzzy matching
        self.gov_colleges = [
            "Ghani Khan Choudhury Institute Of Engineering & Technology, Malda",
            "Alipurduar Government Engineering and Management College, Alipurduar",
            "Cooch Behar Government Engineering College, Cooch Behar",
            "Government College Of Engineering And Leather Technology, Kolkata",
            "Govt. College Of Engg. & Textile Technology, Berhampore",
            "Govt. College Of Engineering & Ceramic Technology, Kolkata",
            "Govt. College Of Engineering & Textile Technology, Serampore",
            "Jalpaiguri Government Engineering College, Jalpaiguri",
            "Kalyani Government Engineering College, Kalyani, Nadia",
            "Ramkrishna Mahato Government Engineering College, Purulia",
            "Institute of Pharmacy, Jalpaiguri"
        ]
        self.gov_keywords = [
            "government", "govt", "university of calcutta", "calcutta university",
            "jadavpur university", "presidency university", "makaut", "wbut",
            "kalyani university", "burdwan university", "vidyasagar university",
            "north bengal university", "west bengal state university", "aliah university"
        ]

        # Normalize lists once at initialization
        self.gov_colleges = [self.normalize_name(c) for c in self.gov_colleges]
        self.gov_keywords = [self.normalize_name(k) for k in self.gov_keywords]

    def normalize_name(self, name: str) -> str:
        """Convert to lowercase, remove non-alpha except spaces, normalize spaces."""
        name = str(name).lower()
        name = re.sub(r'[^a-z\s]', '', name)  # remove all non-letters except spaces
        name = re.sub(r'\s+', ' ', name).strip()  # collapse multiple spaces
        return name

    def classify_college(self, name: str) -> str:
        """Classify a single college name using fuzzy matching and keywords."""
        norm_name = self.normalize_name(name)

        # Exact or fuzzy match with known government colleges
        for gov_college in self.gov_colleges:
            if fuzz.ratio(norm_name, gov_college) >= self.threshold:
                return "Government"

        # Check if any government keyword exists in the normalized name
        if any(k in norm_name for k in self.gov_keywords):
            return "Government"

        return "Private"

    def classify_dataframe(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply classification to a dataframe."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")

        df[column] = df[column].fillna("").astype(str)
        df["college_type"] = df[column].apply(self.classify_college)
        #After evrything some colleges were still miss-classifying but of wrong data input(spelling mistakes)
        df.loc[df['College_Name'] == "UNIVERSITY OF KALYANI KALYANI", 'college_type'] = "Government"
        df.loc[df['College_Name'] == "University of Kalyani Science Instrumentation Centre", 'college_type'] = "Government"
        df.loc[df['College_Name'] == "Maulana Abul Kalam Azad University of Technology West Bengal", 'college_type'] = "Government"
        df.loc[df['College_Name'] == "Bidhan Chandra Krishi Viswa Vidyalaya Mohanpur Nadia", 'college_type'] = "Government"
        return df
