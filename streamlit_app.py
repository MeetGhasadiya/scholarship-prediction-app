import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model & columns
try:
    model = pickle.load(open("scholarship_model.pkl","rb"))
    columns = pickle.load(open("columns.pkl","rb"))
except FileNotFoundError:
    st.error("Model or columns file not found! Please ensure scholarship_model.pkl and columns.pkl exist.")
    st.stop()

st.title("Scholarship Eligibility Prediction System")

# VALIDATION: Define valid categories based on training data columns
# Extract actual categories from the column names
valid_categories = {
    "Education Qualification": [],
    "Gender": [],
    "Community": [],
    "Religion": [],
    "Exservice-men": [],
    "Disability": [],
    "Sports": [],
    "India": []
}

# Parse the columns to extract valid categories
for col in columns:
    for key in valid_categories.keys():
        if col.startswith(key + "_"):
            category_value = col.replace(key + "_", "")
            if category_value not in valid_categories[key]:
                valid_categories[key].append(category_value)

# Note: "Name" columns from training data represent scholarship types
# We don't collect this input - the model predicts general eligibility

st.sidebar.markdown("### Valid Input Categories (from training data)")
with st.sidebar.expander("See training data categories"):
    for key, values in valid_categories.items():
        st.write(f"**{key}:** {', '.join(sorted(values))}")

# User Inputs
education = st.selectbox("Education Qualification", valid_categories["Education Qualification"])
gender = st.selectbox("Gender", valid_categories["Gender"])
community = st.selectbox("Community", valid_categories["Community"])
religion = st.selectbox("Religion", valid_categories["Religion"])
exservice = st.selectbox("Exservice-men", valid_categories["Exservice-men"])
disability = st.selectbox("Disability", valid_categories["Disability"])
sports = st.selectbox("Sports", valid_categories["Sports"])
india = st.selectbox("Indian Citizen", valid_categories["India"])

annual = st.number_input("Annual Percentage", min_value=0.0, max_value=100.0)
income = st.number_input("Annual Income (in Rupees)", min_value=0)

# AUTO-CALCULATION: Extract Annual Percentage & Income ranges from training data
percentage_ranges = []
income_ranges = []

for col in columns:
    if col.startswith("Annual-Percentage_"):
        percentage_ranges.append(col.replace("Annual-Percentage_", ""))
    elif col.startswith("Income_"):
        income_ranges.append(col.replace("Income_", ""))

percentage_ranges.sort()
income_ranges.sort()

st.info("💡 Annual percentage and income are automatically matched to training data ranges")

# AUTO-CALCULATION FUNCTION: Bin Annual Percentage
def get_percentage_bin(percentage):
    """Automatically categorize percentage into training data bins"""
    for bin_range in percentage_ranges:
        if "-" in bin_range:
            parts = bin_range.split("-")
            lower = float(parts[0])
            upper = float(parts[1])
            if lower <= percentage <= upper:
                return bin_range
    return None

# AUTO-CALCULATION FUNCTION: Bin Income
def get_income_bin(income_value):
    """Automatically categorize income into training data bins"""
    # Parse income ranges and match
    for bin_range in income_ranges:
        if "Upto" in bin_range:
            # "Upto 1.5L" means <= 150000
            amount = float(bin_range.replace("Upto ", "").replace("L", "")) * 100000
            if income_value <= amount:
                return bin_range
        elif "to" in bin_range:
            # "1.5L to 3L" means 150000 to 300000
            parts = bin_range.split(" to ")
            lower = float(parts[0].replace("L", "")) * 100000
            upper = float(parts[1].replace("L", "")) * 100000
            if lower <= income_value <= upper:
                return bin_range
        elif "Above" in bin_range:
            # "Above 6L" means > 600000
            amount = float(bin_range.replace("Above ", "").replace("L", "")) * 100000
            if income_value > amount:
                return bin_range
    return None

# AUTO-CALCULATION: Get the bin for user inputs
percentage_bin = get_percentage_bin(annual)
income_bin = get_income_bin(income)

# Display calculated bins to user
col1, col2 = st.columns(2)
with col1:
    if percentage_bin:
        st.success(f"✅ Percentage Category: **{percentage_bin}**")
    else:
        st.warning(f"⚠️ Annual percentage {annual}% is outside training data range")

with col2:
    if income_bin:
        st.success(f"✅ Income Category: **{income_bin}**")
    else:
        st.warning(f"⚠️ Income ₹{income:,} is outside training data range")

# Create input dataframe with ALL columns from training set
# Initialize all columns to 0 (important for one-hot encoding)
input_df = pd.DataFrame(0, index=[0], columns=columns)

# Mapping: Map user inputs to exact column names
categorical_mappings = {
    "Education Qualification_" + education: 1,
    "Gender_" + gender: 1,
    "Community_" + community: 1,
    "Religion_" + religion: 1,
    "Exservice-men_" + exservice: 1,
    "Disability_" + disability: 1,
    "Sports_" + sports: 1,
    "India_" + india: 1,
}

# AUTO-ADD: Include percentage and income bins if they exist
if percentage_bin:
    categorical_mappings["Annual-Percentage_" + percentage_bin] = 1
if income_bin:
    categorical_mappings["Income_" + income_bin] = 1

# Set one-hot encoded values
missing_columns = []
for col_name, value in categorical_mappings.items():
    if col_name in input_df.columns:
        input_df[col_name] = value
    else:
        missing_columns.append(col_name)

# VALIDATION: Check if all required columns were found
if missing_columns:
    st.warning(f"⚠️ WARNING: The following categories were not found in training data: {missing_columns}")
    st.warning("The model may produce inaccurate predictions.")

# VALIDATION: Check data integrity before prediction
if st.button("Predict"):
    try:
        # Verify input shape matches model expectation
        if input_df.shape[1] != len(columns):
            st.error(f"Data shape mismatch! Expected {len(columns)} columns, got {input_df.shape[1]}")
            st.stop()
        
        # Make prediction
        pred = model.predict(input_df)
        prob = model.predict_proba(input_df)
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        eligibility_prob = round(prob[0][1] * 100, 2)
        ineligibility_prob = round(prob[0][0] * 100, 2)
        
        if pred[0] == 1:
            st.success("✅ **Eligible for Scholarship**")
        else:
            st.error("❌ **Not Eligible for Scholarship**")
        
        # Display probability with confidence indicator
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Eligibility Probability", f"{eligibility_prob}%")
        with col2:
            st.metric("Ineligibility Probability", f"{ineligibility_prob}%")
        with col3:
            # Model confidence (how certain the model is)
            confidence = max(eligibility_prob, ineligibility_prob)
            if confidence >= 90:
                st.metric("Model Confidence", f"{confidence}%", delta="Very High")
            elif confidence >= 75:
                st.metric("Model Confidence", f"{confidence}%", delta="High")
            elif confidence >= 60:
                st.metric("Model Confidence", f"{confidence}%", delta="Moderate")
            else:
                st.metric("Model Confidence", f"{confidence}%", delta="Low")
        
        # Key factors affecting eligibility
        st.markdown("---")
        st.subheader("📊 Key Factors in Your Application")
        
        factor_col1, factor_col2 = st.columns(2)
        
        with factor_col1:
            st.markdown("**Positive Indicators:**")
            positive_factors = []
            
            if annual >= 80:
                positive_factors.append("🟢 High academic percentage (≥80%)")
            elif annual >= 70:
                positive_factors.append("🟡 Good academic percentage (≥70%)")
            
            if income <= 150000:
                positive_factors.append("🟢 Low income bracket (financial need)")
            
            if disability == "Yes":
                positive_factors.append("🟢 Disability category eligible")
            
            if sports == "Yes":
                positive_factors.append("🟢 Sports quota eligible")
            
            if exservice == "Yes":
                positive_factors.append("🟢 Ex-servicemen category")
            
            if community in ["SC/ST", "OBC", "Minority"]:
                positive_factors.append(f"🟢 Reserved category ({community})")
            
            if education in ["Postgraduate", "Doctrate"]:
                positive_factors.append(f"🟢 Higher education ({education})")
            
            if positive_factors:
                for factor in positive_factors:
                    st.write(factor)
            else:
                st.write("🟡 No strong positive indicators found")
        
        with factor_col2:
            st.markdown("**Areas of Concern:**")
            concern_factors = []
            
            if annual < 60:
                concern_factors.append("🔴 Percentage below typical threshold (<60%)")
            elif annual < 70:
                concern_factors.append("🟡 Percentage in lower range (<70%)")
            
            if income > 600000:
                concern_factors.append("🟡 Income above typical scholarship limits")
            
            if india == "Out":
                concern_factors.append("🔴 Non-Indian citizen (limited eligibility)")
            
            if disability == "No" and sports == "No" and exservice == "No" and community == "General":
                concern_factors.append("🟡 No special category advantages")
            
            if concern_factors:
                for factor in concern_factors:
                    st.write(factor)
            else:
                st.write("✅ No major concerns detected")
        
        # Improvement suggestions (only if not eligible)
        if pred[0] == 0:
            st.markdown("---")
            st.subheader("💡 Suggestions to Improve Eligibility")
            suggestions = []
            
            if annual < 70:
                suggestions.append("📚 **Improve academic performance** - Aim for ≥70% to increase chances")
            
            if income > 300000 and annual < 80:
                suggestions.append("🎓 **Focus on academic excellence** - Higher percentages can compensate for income bracket")
            
            if sports == "No" and disability == "No":
                suggestions.append("🏅 **Explore special categories** - Sports achievements or other recognitions can help")
            
            suggestions.append("📋 **Apply to multiple scholarships** - Different schemes have different criteria")
            suggestions.append("🔍 **Look for category-specific scholarships** - Some are designed for your specific profile")
            
            for suggestion in suggestions:
                st.info(suggestion)
        
        # Show input summary
        st.markdown("---")
        st.subheader("📋 Your Application Summary")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown("**Personal Details:**")
            st.write(f"• **Education:** {education}")
            st.write(f"• **Gender:** {gender}")
            st.write(f"• **Community:** {community}")
            st.write(f"• **Religion:** {religion}")
            st.write(f"• **Citizenship:** {india}")
        
        with summary_col2:
            st.markdown("**Academic & Financial:**")
            st.write(f"• **Annual Percentage:** {annual}%")
            if percentage_bin:
                st.write(f"  → Binned as: {percentage_bin}")
            st.write(f"• **Annual Income:** ₹{income:,}")
            if income_bin:
                st.write(f"  → Binned as: {income_bin}")
            st.write(f"• **Exservice-men:** {exservice}")
            st.write(f"• **Disability:** {disability}")
            st.write(f"• **Sports:** {sports}")
        
        # Model information
        st.markdown("---")
        st.caption("ℹ️ **About this prediction:** This system uses a Random Forest machine learning model trained on historical scholarship data. Predictions are based on patterns in past applications and should be used as guidance only.")
        
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.error("Please check your inputs and try again.")
