#IMPORTING REQUIRED MODULES
import pandas as pd
import re

#CREATING A DATA PREPROCESSING CLASS
class Data_Mapping:
    def __init__(self):
        pass
    #CREATING A PREPROCESSING FUNCTIONM
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        #1st Remove Un-necessary Spaces From Column Names
        df.columns = df.columns.str.strip()

        # 2nd Rename the columns (standardizing column names)
        rename_map = {
            "Institute": "College_Name",
            "Program": "Branch",
            "Quota": "Domicile",
            "Category": "Reservation"
        }
        #Applying it to the data-frame
        df = df.rename(mapper=rename_map, axis=1)

        # 3. Drop unnecessary columns
        for col in ["Sr.No", "Stream"]:
            if col in df.columns:
                df = df.drop([col], axis=1)

        # 4. Clean Branch column
        if "Branch" in df.columns:
            df = self._clean_branch(df, column="Branch")

        # 5. Clean Round
        if "Round" in df.columns:
            df["Round"] = df["Round"].fillna("").str.replace("Round", "", regex=False).str.strip()

        # 6. Clean College_Name
        if "College_Name" in df.columns:
            df["College_Name"] = df["College_Name"].fillna("").str.replace(",", "", regex=False).str.strip()

        # 7. Ensure numeric ranks
        for col in ["Opening Rank", "Closing Rank"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _clean_branch(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Internal method to clean and standardize branch names."""

        new_list = df[column].fillna("").astype(str)

        # Mapping function for known branch patterns
        def map_value(val):
            val_lower = val.lower()
            if re.search(r"\bartificial intelligence\b", val_lower):
                return "AI"
            elif re.search(r"\bmachine learning\b", val_lower):
                return "AI"
            elif re.search(r"\bcomputer science\b", val_lower):
                return "CSE"
            elif re.search(r"\biot\b|internet of things\b", val_lower):
                return "IOT"
            elif re.search(r"\bbiotech\b|biotechnology\b", val_lower):
                return "BIO-TECH"
            elif re.search(r"\belectronics\b", val_lower):
                return "ECE"
            elif re.search(r"\bcivil\b", val_lower):
                return "CIVIL"
            elif re.search(r"\bmechanical\b", val_lower):
                return "MECHANICAL"
            elif re.search(r"\bchemical\b", val_lower):
                return "CHEMICAL"
            elif re.search(r"\bproduction\b", val_lower):
                return "PRODUCTION"
            elif re.search(r"\binformation\b", val_lower):
                return "IT"
            elif re.search(r"\belectrical\b", val_lower):
                return "EE"
            else:
                return "Other"  # return original if no match

        # Apply mapping to create a "_short" column
        df[column + "_short"] = new_list.apply(map_value)

        # Remove unwanted characters
        df[column] = (
            df[column]
            .fillna("")
            .str.replace("TFW", "", regex=False)
            .str.replace("Tfw", "", regex=False)
            .str.replace("()", "", regex=False)
            .str.replace("-", " ", regex=False)
            .str.replace(",", " ", regex=False)
            .str.strip()
        )
        df[column] = df[column].str.extract(r"\(\s*(.*?)\s*\)")[0].fillna(df[column]).str.strip()
        df[column] = df[column].str.replace(r"[^a-zA-Z\s]", "", regex=True)
        #Don't need this column closing rank is enough
        df= df.drop(['Opening Rank'], axis=1)
        #Also Seat Type is redundant Qouta+Reservation already implies seat type
        df= df.drop(['Seat Type'], axis=1)
        return df
