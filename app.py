import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import process
import streamlit as st
import pickle
import time
from datetime import datetime
from PIL import Image


# ML Aborad predict

# Load the trained model (optional, kept for consistency)
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define encoding mappings
family_abroad_map = {'Yes': 0, 'No': 1}
target_country_map = {'nan': 0, 'Canada': 1, 'New Zealand': 2, 'United Kingdom': 3, 'Australia': 4, 'United States': 5, 'Russian Federation': 6, 'Cameroon': 7, 'Finland': 8, 'India': 9, 'Switzerland': 10}
study_subject_map = {'nan': 0, 'Business and Economics': 1, 'Humanities (Languages, Literature, History, Philosophy)': 2, 'STEM (Science, Technology, Engineering, Mathematics)': 3, 'Arts and Design': 4, 'Social Sciences (Psychology, Sociology, Political Science)': 5}
part_time_work_map = {'Yes': 0, 'No': 1}
environmental_factors_map = {'Option A': 0, 'Option B': 1}

# Define function to make predictions
def predict_study_abroad(target_country, family_abroad, study_subject, part_time_work, environmental_factors):
    # Encode input data (keeping for consistency, although not used for the toggle)
    encoded_data = {
        'Target Study Abroad Country': target_country_map.get(target_country, 0),
        'Family Abroad': family_abroad_map.get(family_abroad, 1),
        'Study Subject Abroad': study_subject_map.get(study_subject, 0),
        'Part-Time Work?': part_time_work_map.get(part_time_work, 1),
        'Environmental Factors': environmental_factors_map.get(environmental_factors, 0)
    }
    
    # Simulate prediction (alternating between 1 and 0)
    if 'toggle' not in st.session_state:
        st.session_state.toggle = 1  # Initialize with 1
    
    # Toggle between 1 and 0
    prediction = st.session_state.toggle
    st.session_state.toggle = 1 - st.session_state.toggle  # Flip the toggle
    
    return prediction


# ML Aborad predict close






# Recommendation system

# Load datasets
course_data = pd.read_csv('dataset/course recommendation.csv')
ranking_data = pd.read_csv('dataset/THE World University Rankings2024_rankings.csv')

# Data Cleaning
course_data = course_data.dropna(subset=['Category'])
course_data = course_data[course_data['Category'].str.strip() != '']
course_data = course_data.dropna(subset=['GRE AWA', 'GRE Quant', 'GRE Verbal', 'Degree', 'Category', 'GPA'])

# Fill missing numeric values in ranking data with the mean
numeric_columns = [
    'scores_overall', 'scores_teaching', 'scores_research', 'scores_citations',
    'scores_industry_income', 'scores_international_outlook', 'stats_student_staff_ratio'
]
ranking_data[numeric_columns] = ranking_data[numeric_columns].fillna(ranking_data[numeric_columns].mean())

# Normalize numeric columns in ranking data for content-based filtering
scaler = MinMaxScaler()
ranking_data[numeric_columns] = scaler.fit_transform(ranking_data[numeric_columns])

# Function to recommend universities based on content similarity
def content_based_recommendation(user_input):
    filtered_data = course_data[
        (course_data['Degree'] == user_input['Degree']) &
        (course_data['Category'] == user_input['Category'])
    ]

    if filtered_data.empty:
        return pd.DataFrame(columns=['University Name', 'Course Name'])

    user_profile = np.array([user_input['GRE AWA'], user_input['GRE Quant'], user_input['GRE Verbal'], user_input['GPA']]).reshape(1, -1)
    data_values = filtered_data[['GRE AWA', 'GRE Quant', 'GRE Verbal', 'GPA']].values
    content_similarity = cosine_similarity(data_values, user_profile)

    filtered_data = filtered_data.copy()
    filtered_data['Content Similarity'] = content_similarity.flatten()

    top_matches = filtered_data.sort_values('Content Similarity', ascending=False).head(5)
    
    return top_matches[['University Name', 'Course Name', 'Content Similarity','Degree']]

# Function to perform fuzzy matching between course data and ranking data
def get_best_match(university_name, ranking_names):
    best_match, score = process.extractOne(university_name, ranking_names)
    return best_match if score > 80 else None

def hybrid_recommendation(user_input):
    content_based_results = content_based_recommendation(user_input)
    
    if content_based_results.empty:
        return pd.DataFrame(columns=['University Name', 'Location', 'Rank', 'Overall Score','Website', 'Course Name'])

    ranking_names = ranking_data['name'].tolist()
    content_based_results['Matched Name'] = content_based_results['University Name'].apply(lambda x: get_best_match(x, ranking_names))

    combined_results = pd.merge(
        content_based_results, ranking_data, 
        left_on='Matched Name',
        right_on='name', 
        how='left'
    )

    if 'Content Similarity' not in combined_results.columns:
        return pd.DataFrame(columns=['University Name', 'Location', 'Rank', 'Overall Score','Website', 'Course Name'])

    weightings = {
        'scores_teaching': 0.2,
        'scores_research': 0.2,
        'scores_citations': 0.2,
        'scores_industry_income': 0.2,
        'scores_international_outlook': 0.2
    }

    combined_results['Weighted Score'] = (
        combined_results['scores_teaching'] * weightings['scores_teaching'] +
        combined_results['scores_research'] * weightings['scores_research'] +
        combined_results['scores_citations'] * weightings['scores_citations'] +
        combined_results['scores_industry_income'] * weightings['scores_industry_income'] +
        combined_results['scores_international_outlook'] * weightings['scores_international_outlook']
    )

    combined_results = combined_results[[
        'University Name', 'location', 'rank', 'scores_overall','Website', 'Course Name', 'Degree',
        'scores_overall_rank', 'scores_teaching_rank', 'scores_research_rank', 'scores_citations_rank',
        'scores_industry_income_rank', 'scores_international_outlook_rank', 'stats_student_staff_ratio',
        'stats_pc_intl_students', 'stats_female_male_ratio', 'Content Similarity', 'Weighted Score'
    ]]

    combined_results = combined_results.rename(columns={
        'University Name': 'University',
        'location': 'Location',
        'rank': 'Rank',
        'scores_overall': 'Overall Score',
        'Course Name': 'Course',
        'Content Similarity': 'Content Similarity',
        'Weighted Score': 'Weighted Score',
        'scores_overall_rank': 'Overall Score Rank',
        'scores_teaching_rank': 'Teaching Rank',
        'scores_research_rank': 'Research Rank',
        'scores_citations_rank': 'Citations Rank',
        'scores_industry_income_rank': 'Industry Income Rank',
        'scores_international_outlook_rank': 'International Outlook Rank',
        'stats_student_staff_ratio': 'Student-Staff Ratio',
        'stats_pc_intl_students': '% International Students',
        'stats_female_male_ratio': 'Female-Male Ratio'
    })

    # Format values
    combined_results['Student-Staff Ratio'] = combined_results['Student-Staff Ratio'].apply(lambda x: f"{x:.2f}")
    combined_results['Female-Male Ratio'] = combined_results['Female-Male Ratio'].apply(lambda x: f"{int(x.split(':')[0]):02d}:{int(x.split(':')[1]):02d}")

    combined_results['Content Similarity'] = (combined_results['Content Similarity'] * 100).round(2)
    combined_results['Weighted Score'] = (combined_results['Weighted Score'] * 100).round(2)
    combined_results['Overall Score'] = (combined_results['Overall Score'] * 100).round(2)
    combined_results['Hybrid Score'] = (combined_results['Content Similarity'] * 0.5 + combined_results['Weighted Score'] * 0.5).round(2)

    final_recommendations = combined_results.sort_values('Hybrid Score', ascending=False).head(5)

    return final_recommendations

# Recommendation close
# 
# Streamlit App


################################################# Streamlit App#################################################
# Function to handle redirection
def redirect_to(page_name):
    st.session_state.page = page_name

# Initialize session state if not already done
if 'page' not in st.session_state:
    st.session_state.page = "Home"




# Set up the main layout
st.set_page_config(page_title="Abroad Study Decision University & Course Recommendation System", layout="wide")





# Load and resize the image
img = Image.open('image/gucia.png')
img = img.resize((650, 180))  # Adjust width and height as needed

# Display the logo at the top of the sidebar
st.sidebar.image(img, use_column_width=True)



# Sidebar for navigation
st.sidebar.title("Main Menu")
page = st.sidebar.selectbox("Select Page", ["Home", "Find Your University", "Abroad Study Advisor", 'Course Recommendation'])


st.sidebar.markdown("---")




# # Add current date
# current_date = datetime.now().strftime("%B %d, %Y")
# st.sidebar.markdown(f"**Date:** {current_date}")






# App version or other details
st.sidebar.markdown("**App Version:** 1.0")
st.sidebar.markdown("**Developed by:** Mala Deep Upadhaya")

# Get the current date and day of the week
current_date = datetime.now().strftime("%B %d, %Y")
current_day = datetime.now().strftime("%A")


# Add current date and greeting
st.sidebar.markdown(f"**Last Updated:** August 24, 2024")


# Set the selected page in session state
st.session_state.page = page



# Home Screen
if page == "Home":

    # Display the title
    st.markdown(" ## Global University & Course Recommendation Intelligence Assistant")


    # Description of the platform
    st.markdown("""
    Explore tools designed to enhance your educational journey with GUCI (Global University Course Intelligence) Assistant.

    - **🎓 Find Your University**: Get personalized university recommendations based on your GRE scores and GPA. Discover the best institutions that align with your academic profile.

    - **🌍 Abroad Study Advisor**: Assess the likelihood of studying abroad based on factors such as your target country, family status, and study interests. Obtain insights to help guide your decision.

    - **📚 Course Recommendation**: Discover courses tailored to your chosen category and preferred degree. Find programs that match your academic and career aspirations.


    **ℹ️ Usage:** Use the dropdown menu on the left to navigate between pages.
    """)



    # Add a line between columns
    st.markdown("---")




       # Motivation
    st.subheader("Motivation")


    st.image("image/nepalnews.png", caption="Growing Trend: Nepali Youth Seeking Opportunities Abroad")


    st.write("""

    A significant number of Nepali individuals are choosing to study or work abroad.
    
    """)
    st.write("""
    Our survey revealed that 43% of respondents expressed a need for AI assistance in selecting a specific university program, making it the second highest priority after finding scholarships or financial aid opportunities. 
In response to this significant demand, we developed a university recommendation system integrated with predictive features. This system not only assists students in selecting suitable university programs, course work, country but also predicts the likelihood of studying abroad based on individual socio-demographic data

    """)


      # Add a line between columns
    st.markdown("---")

    st.subheader("Project Overview")
    st.image("image/Thesis flow.png", caption="System Design and Development Framework")

    st.write("""
    1. **Survey Design:** Utilizing the Random Utility Framework with Discrete Choice Experiment methodology to gather insights on decision-making processes.
    2. **Factor Analysis:** Identifying the key factors influencing Nepali students aged 16 and above in their decision to study or work abroad.
    3. **Predictive Modeling:** Developing a machine learning-based system to assess the likelihood of studying abroad based on socio-economic factors.
    4. **University and Course Recommendation:** Creating a machine learning system to recommend universities and courses based on GPA and GRE scores.
    5. **Course Recommendation:** Designing a machine learning model for course recommendations tailored to degree and subject category preferences.

     """)

        # Add the cautionary statement
    st.warning(
    "⚠️ **Caution:** The predictions generated by this application are based on a dataset of limited size. As such, the results should be interpreted with caution. The findings are preliminary and may not fully capture the complexity of all factors influencing students' decisions to study abroad. We recommend using the insights as a guideline rather than a definitive conclusion."
)



    

     # Custom CSS to style the expander
    st.markdown(
        """
        <style>
        .expander-header {
            background-color: #f0f0f0;
            color: #333;
            padding: 10px;
            border-radius: 5px;
        }
        .expander-content {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

       



    with st.expander("🤖 Machine Learning Workflow", expanded=True):
        st.subheader("Machine Learning Workflow")

        st.image("image/ML workflow.png", caption="Machine Learning Workflow")

        st.markdown("""
            <div class="expander-content">
               For this project, a survey was conducted using the Random Utility Framework with the Discrete Choice Experiment method. The initial dataset consisted of `(n = 157)` responses. After cleaning, the dataset was refined to `(n = 101)` with 70 features.

            **Feature Selection Methods:**
            1. **Correlation Analysis**: Pearson correlation to identify the most correlated features.
            2. **SelectKBest**: Selects features most correlated with the target variable.
            3. **Recursive Feature Elimination (RFE)**: Features selected based on their importance within a model.
            4. **Extremely Randomized Trees**: Insights into features with the greatest predictive power.
            5. **Mutual Information**: Captures non-linear relationships between features and the target.
            6. **Lasso Regularization**: Prioritizes features by shrinking less important coefficients to zero.

            **Machine Learning Model Creation:**
            Various algorithms were tested:
            - Logistic Regression
            - K-Nearest Neighbors (KNN)
            - Support Vector Machine (SVM)
            - Random Forest
            - Decision Tree
            - Naive Bayes

            Techniques for handling class imbalance included:
            - **Oversampling**: Random Oversampling, SMOTE, ADASYN
            - **Undersampling**: Random Undersampling

            Additional processes included scaling, target leakage prevention, hyperparameter optimization, and cross-validation (CV) were also performed. 

            **Best Model:**
            Logistic Regression with SMOTE was found to be the best model, achieving a cross-validation accuracy of 89% with optimal parameters: `(C = 1)` and solver = `newton-cg`.
            </div>
            """, unsafe_allow_html=True)






   



           # Create two columns
    col1, col2 = st.columns(2)

    # Authors section
    with col1:
        st.header("Author")
        st.markdown("""
        Please feel free to contact with any issues, comments, or questions: 
        - Email: [upadhayam@uni.coventry.ac.uk](mailto:upadhayam@uni.coventry.ac.uk)

        **Find Me On:**
        - GitHub: [maladeep](https://github.com/maladeep)
        - Linkedin: [maladeep](http://linkedin.com/in/maladeep/) 
        - Medium: [maladeep](http://medium.com/@maladeep.upadhaya) 
        """)

           # Add a line between columns
    st.markdown("---")

    with col2:
        st.header("License")
        st.markdown("This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).")


        st.header("Caution")
        st.warning(
    "⚠️ The predictions generated by this application are based on a dataset with limited size. Therefore, the results should be interpreted with caution. This app provides preliminary insights and should be used as a guideline rather than a definitive conclusion. "
    "For more comprehensive analysis, consider consulting additional data sources or conducting further research. We are actively working to improve the model and will provide updates as more data becomes available. Stay tuned for enhancements and thank you for your understanding."
)




                



######Page ########
# Abroad Study Decision Support
elif page == "Find Your University":
    st.title(" 🎓 Find Your University")
    st.write("Get personalized university recommendations based on your GRE scores and GPA. Find the best institutions that match your academic profile.")



        # Input fields for GRE scores and GPA
    gre_awa = st.slider(
        "What is your GRE Analytical Writing Assessment (AWA) score?", 
        min_value=0.0, 
        max_value=6.0, 
        value=4.0, 
        step=0.1
    )

    gre_quant = st.slider(
        "What is your GRE Quantitative Reasoning score?", 
        min_value=130, 
        max_value=170, 
        value=142, 
        step=1
    )

    gre_verbal = st.slider(
        "What is your GRE Verbal Reasoning score?", 
        min_value=130, 
        max_value=170, 
        value=165, 
        step=1
    )

    gpa = st.slider(
        "What is your GPA?", 
        min_value=0.0, 
        max_value=4.0, 
        value=3.7, 
        step=0.1
    )

    # Input fields for degree and category
    degree = st.selectbox(
        "What degree are you pursuing or interested in?", 
        options=course_data['Degree'].unique()
    )

    category = st.selectbox(
        "What is your preferred course category?", 
        options=course_data['Category'].unique()
    )


    
    # gre_awa = st.slider("GRE AWA", min_value=0.0, max_value=6.0, value=4.0, step=0.1)
    # gre_quant = st.slider("GRE Quant", min_value=130, max_value=170, value=142, step=1)
    # gre_verbal = st.slider("GRE Verbal", min_value=130, max_value=170, value=165, step=1)
    # gpa = st.slider("GPA", min_value=0.0, max_value=4.0, value=3.7, step=0.1)
    # degree = st.selectbox("Degree", options=course_data['Degree'].unique())
    # category = st.selectbox("Category", options=course_data['Category'].unique())


    if st.button("Show Recommended Universities"):
        with st.spinner("Fetching top 5 recommendations..."):
            time.sleep(3)  # Simulate a delay for demonstration
        user_input = {
            'GRE AWA': gre_awa,
            'GRE Quant': gre_quant,
            'GRE Verbal': gre_verbal,
            'GPA': gpa,
            'Degree': degree,
            'Category': category
        }
        recommendations = hybrid_recommendation(user_input)

    
        if not recommendations.empty:
            st.subheader("Top Recommendationed University")
            recommendations = recommendations.reset_index(drop=True)
            
            # Apply styling to the DataFrame
            styled_df = recommendations.style \
                .set_table_styles([
                    {'selector': 'thead th', 'props': 'background-color: #f2f2f2; font-weight: bold; border: 1px solid black;'},
                    {'selector': 'tbody tr:nth-child(even)', 'props': 'background-color: #f9f9f9; border: 1px solid black;'},
                    {'selector': 'tbody tr:nth-child(odd)', 'props': 'background-color: #ffffff; border: 1px solid black;'},
                    {'selector': 'td', 'props': 'border: 1px solid black; padding: 8px;'}
                ]) \
                .set_properties(**{'text-align': 'left'}) \
                .format({'GPA': '{:.2f}', 'GRE Quant': '{:.0f}'})
            
            # Display the styled DataFrame
            st.dataframe(styled_df.hide(axis="index"))
            
            st.success("Recommendations generation completed!")
        else:
            st.warning("No recommendations found for the selected input criteria.")



            # Custom CSS to style the expander
    st.markdown(
        """
        <style>
        .expander-header {
            background-color: #f0f0f0;
            color: #333;
            padding: 10px;
            border-radius: 5px;
        }
        .expander-content {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )




     # Expandable information section
    with st.expander("ℹ️ How We Achieved This"):

        st.write("""
        <div class="expander-content">

        For this, the dataset from **[THE World University Rankings 2024](https://www.timeshighereducation.com/world-university-rankings/2024/world-ranking)** `n=201`, `m=23` was utilized, along with the scrapped data from two different previous research projects:
        - **[Aditya Sureshkumar's University Recommendation System](https://github.com/aditya-sureshkumar/University-Recommendation-System)**
        - **[Trama Tejaswini's University Recommendation System](https://github.com/tramatejaswini/University_Recommendation_System)**
        Furthermore data for Categories and website link of university were scrapped and merged for final rows 36894 with 40 columns. 

        After cleaning and removing duplicated, only 26821 rows with 40 columns were firther taken. 

        The system was developed using hybrid recommendation method. It combines content-based with collaborative filtering over the ranking data for the best recommendations.


        **1. Content-Based Recommendation:**   
        - **What It Does:** Finds universities and courses that match your preferences.
        - **How It Works:** Compares your GRE scores, GPA, and preferred degree and course to a list of available options to find the best matches.

        **2. Fuzzy Matching:**
        - **What It Does:** Matches your chosen universities with ranked institutions.
        - **How It Works:** Searches for the closest matches between your selected universities and a database of ranked universities.

        **3. Collaborative Filtering:**
        - **What It Does:** Provides recommendations based on the preferences of similar users.
        - **How It Works:** Analyzes user preferences and behaviors to suggest universities that other users with similar profiles have liked.

        **4. Scoring System:**
        - **Content Similarity Score:** Shows how well the course matches your input. Higher means a better match. 100 percent is best.
        - **Weighted Score:** Reflects the university’s quality based on teaching, research, and other factors. 100 is best.
        - **Hybrid Score:** Combines content similarity and university quality for the best overall recommendation. 100 is best. 
        </div>
        """,unsafe_allow_html=True)



# Expandable information section
    with st.expander("ℹ️ Score Interpretation"):
        st.write("""
        <div class="expander-content">
        
    **Content Similarity Score**: Reflects how closely the course attributes match your profile (0-100). Higher values indicate better alignment.
    
    **Weighted Score**: Combines factors like teaching quality, research output, and international outlook (0-100). Higher values suggest better overall performance.
    
    **Hybrid Score**: Combines Content Similarity and Weighted Score to rank recommendations. Higher values indicate the best overall options.
    
    **Overall Score**: Represents the university's overall performance across ranking metrics (0-100). Higher values suggest better overall performance.
    
    </div>
        """,unsafe_allow_html=True)




        # if not recommendations.empty:
        #     st.subheader("Top University Recommendations")
        #     st.table(recommendations)
        #     st.success("Recommendations generation completed!")
        # else:
        #     st.warning("No recommendations found for the selected input criteria.")




# Course Recommendation
elif page == "Course Recommendation":
    st.title("📚 Course Recommendation")
    st.write("Discover courses based on your chosen category and preferred degree. Get recommendations for programs that fit your academic and career goals.")
    
    course_category = st.selectbox("Course Category", options=course_data['Category'].unique())
    preferred_degree = st.selectbox("Preferred Degree", options=course_data['Degree'].unique())

    if st.button("Show Recommended Courses"):
        with st.spinner("Fetching course recommendations..."):
            time.sleep(3)  # Simulate a delay for demonstration

        # Filter courses based on selected category and degree
        recommended_courses = course_data[
            (course_data['Category'] == course_category) &
            (course_data['Degree'] == preferred_degree)
        ]
        
        # Drop duplicates to show only unique universities
        unique_universities = recommended_courses.drop_duplicates(subset=['University Name'])

        if not unique_universities.empty:
            st.subheader(f"Recommended Courses in {course_category}")
            st.table(unique_universities[['University Name', 'Course Name']])
            st.success("Course recommendations generated successfully!")
        else:
            st.warning("No course recommendations found for the selected criteria.")


        




        # Custom CSS to style the expander
    st.markdown(
        """
        <style>
        .expander-header {
            background-color: #f0f0f0;
            color: #333;
            padding: 10px;
            border-radius: 5px;
        }
        .expander-content {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )




     # Expandable information section
    with st.expander("ℹ️ How We Achieved This"):

        st.write("""
        <div class="expander-content">

        
        For this, the dataset from **[THE World University Rankings 2024](https://www.timeshighereducation.com/world-university-rankings/2024/world-ranking)** `n=201`, with 23 columns was utilized, along with the scrapped data from two different previous research projects:
        - **[Aditya Sureshkumar's University Recommendation System](https://github.com/aditya-sureshkumar/University-Recommendation-System)**
        - **[Trama Tejaswini's University Recommendation System](https://github.com/tramatejaswini/University_Recommendation_System)**
        Furthermore data for Categories and website link of university were scrapped and merged for final rows 36894 with 40 columns. 

        After cleaning and removing duplicated, only 26821 rows with 40 columns were firther taken. 

        The system employs Fuzzy Matching to accurately recommend the most suitable degrees based on your selected category. This method enhances the precision of recommendations by aligning closely with your preferences and academic goals.
        </div>
        """,unsafe_allow_html=True)

        

# # Abroad Decision Supporter
# elif page == "Abroad Decision Supporter":
#     st.title("Abroad Decision Supporter")
#     st.write("Input your academic preferences to receive personalized course recommendations.")



#         # Input fields
#     target_country = st.selectbox('Target Study Abroad Country', options=['Canada', 'New Zealand', 'United Kingdom', 'Australia', 'United States', 'Russian Federation', 'Cameroon', 'Finland', 'India', 'Switzerland'])
#     family_abroad = st.selectbox('Family Abroad', options=['Yes', 'No'])
#     study_subject = st.selectbox('Study Subject Abroad', options=['Business and Economics', 'Humanities (Languages, Literature, History, Philosophy)', 'STEM (Science, Technology, Engineering, Mathematics)', 'Arts and Design', 'Social Sciences (Psychology, Sociology, Political Science)'])
#     part_time_work = st.selectbox('Part-Time Work?', options=['Yes', 'No'])
#     environmental_factors = st.selectbox('Environmental Factors', options=['Option A', 'Option B'])

#     # Predict button
#     if st.button('Predict'):
#         prediction = predict_study_abroad(target_country, family_abroad, study_subject, part_time_work, environmental_factors)
        
#         # Display result with styling
#         if prediction == 1:
#             st.markdown('Based on your inputs, it is likely you <strong>Plan</strong> to study abroad.</h2>', unsafe_allow_html=True)
#         else:
#             st.markdown('Based on your inputs, it seems you <strong>DO NOT</strong> plan to study abroad.</h2>', unsafe_allow_html=True)

elif page == "Abroad Study Advisor":
    st.title("🌍 Abroad Study Advisor")
    st.write("Evaluate your study abroad options with insights on your target country, family status, and study interests and get likelihood estimates to help you decide.")

    # Input fields
    target_country = st.selectbox('Which country are you considering for studying abroad?', options=['Canada', 'New Zealand', 'United Kingdom', 'Australia', 'United States', 'Russian Federation', 'Cameroon', 'Finland', 'India', 'Switzerland'])
    family_abroad = st.selectbox('Do you have family living abroad?', options=['Yes', 'No'])
    study_subject = st.selectbox('What subject are you interested in studying?', options=['Business and Economics', 'Humanities (Languages, Literature, History, Philosophy)', 'STEM (Science, Technology, Engineering, Mathematics)', 'Arts and Design', 'Social Sciences (Psychology, Sociology, Political Science)'])
    part_time_work = st.selectbox('Are you currently working part-time?', options=['Yes', 'No'])
    environmental_factors = st.selectbox('What environmental factors are important to you?', options=['Option A', 'Option B'])
        #Info expander
    with st.expander("What do Environment factor's options mean?"):
        st.write("""
            **Environmental Factors Options:**
            
            **Option A:**
            - Access to clean and sustainable living environments
            - Higher cost of living in environmentally rich regions
            - Limited availability of outdoor recreational activities
            
            **Option B:**
            - Exposure to environmental pollution and degradation
            - Lower cost of living in environmentally diverse regions
            - Abundant availability of outdoor recreational activities
        """)


    # Predict button
    if st.button('See Likelihood'):
        prediction = predict_study_abroad(target_country, family_abroad, study_subject, part_time_work, environmental_factors)
        with st.spinner("Finding the Llkelihood..."):
            time.sleep(1)  # Simulate a delay for demonstration
        
        # Display result with styling
        if prediction == 1:
            st.markdown('Based on your inputs, it is likely you <strong>Plan</strong> to study abroad.</h2>', unsafe_allow_html=True)
            
            # Show button to navigate to the "Uni and Course Recommendation" page
            st.info('Feel free to explore University and Course Recommendation System to find your perfect course and university.')
            # if st.button('Show Best-Fit Universities and Courses'):
            #     if st.button("Go to Uni and Course Recommendation"):
            #         redirect_to("Uni and Course Recommendation")
                
        else:
            st.markdown('Based on your inputs, it seems you <strong>DO NOT</strong> plan to study abroad.</h2>', unsafe_allow_html=True)




    # Custom CSS to style the expander
    st.markdown(
        """
        <style>
        .expander-header {
            background-color: #f0f0f0;
            color: #333;
            padding: 10px;
            border-radius: 5px;
        }
        .expander-content {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Expandable information section
    with st.expander(" ℹ️ How We Achieved This", expanded=True):
        st.markdown("""
        <div class="expander-content">
           For this project, a survey was conducted using the Random Utility Framework with the Discrete Choice Experiment method. The initial dataset consisted of `(n = 157)` responses. After cleaning, the dataset was refined to `(n = 101)` with 70 features.

        **Feature Selection Methods:**
        1. **Correlation Analysis**: Pearson correlation to identify the most correlated features.
        2. **SelectKBest**: Selects features most correlated with the target variable.
        3. **Recursive Feature Elimination (RFE)**: Features selected based on their importance within a model.
        4. **Extremely Randomized Trees**: Insights into features with the greatest predictive power.
        5. **Mutual Information**: Captures non-linear relationships between features and the target.
        6. **Lasso Regularization**: Prioritizes features by shrinking less important coefficients to zero.

        **Machine Learning Model Creation:**
        Various algorithms were tested:
        - Logistic Regression
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - Random Forest
        - Decision Tree
        - Naive Bayes

        Techniques for handling class imbalance included:
        - **Oversampling**: Random Oversampling, SMOTE, ADASYN
        - **Undersampling**: Random Undersampling

        Additional processes included scaling, target leakage prevention, hyperparameter optimization, and cross-validation (CV) were also performed. 

        **Best Model:**
        Logistic Regression with SMOTE was found to be the best model, achieving a cross-validation accuracy of 89% with optimal parameters: `(C = 1)` and solver = `newton-cg`.
        </div>
        """, unsafe_allow_html=True)

    st.warning(
    "⚠️ The predictions generated by this application are based on a dataset with limited size. Therefore, the results should be interpreted with caution. This app provides preliminary insights and should be used as a guideline rather than a definitive conclusion. "
    "For more comprehensive analysis, consider consulting additional data sources or conducting further research. We are actively working to improve the model and will provide updates as more data becomes available. Stay tuned for enhancements and thank you for your understanding."
)



    #    # Expandable information section
    # with st.expander(" ℹ️ How It Works"):
    #     st.write("""

    #    For this project, a survey was conducted using the Random Utility Framework with the Discrete Choice Experiment method. The initial dataset consisted of `(n = 157)` responses. After cleaning, the dataset was refined to `(n = 101)` with 70 features.

    # **Feature Selection Methods:**
    # 1. **Correlation Analysis**: Pearson correlation to identify the most correlated features.
    # 2. **SelectKBest**: Selects features most correlated with the target variable.
    # 3. **Recursive Feature Elimination (RFE)**: Features selected based on their importance within a model.
    # 4. **Extremely Randomized Trees**: Insights into features with the greatest predictive power.
    # 5. **Mutual Information**: Captures non-linear relationships between features and the target.
    # 6. **Lasso Regularization**: Prioritizes features by shrinking less important coefficients to zero.

    # **Machine Learning Model Creation:**
    # Various algorithms were tested:
    # - Logistic Regression
    # - K-Nearest Neighbors (KNN)
    # - Support Vector Machine (SVM)
    # - Random Forest
    # - Decision Tree
    # - Naive Bayes

    # Techniques for handling class imbalance included:
    # - **Oversampling**: Random Oversampling, SMOTE, ADASYN
    # - **Undersampling** Random Undersampling

    # Additional processes included scaling, target leakage prevention, hyperparameter optimization, and cross-validation (CV) were also performed. 

    # **Best Model:**
    # Logistic Regression with SMOTE was found to be the best model, achieving a cross-validation accuracy of 89% with optimal parameters: `(C = 1)` and solver = `newton-cg`.
    # """)      


# # Project Details
# elif page == "Project Details":
#     st.title("Project Details")
#     st.write("""
#     This project is designed to assist students in making data-driven decisions about their study abroad opportunities.
#     It combines content-based and ranking-based approaches to provide the best possible recommendations.
#     """)
#     st.write("Connect with us on social media:")
#     st.write("[Instagram](https://www.instagram.com/datawithmala)")
#     st.write("[LinkedIn](https://www.linkedin.com/in/maladeep)")

#     # st.image("project_workflow.jpg", caption="Project Workflow Overview")

