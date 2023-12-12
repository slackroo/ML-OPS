import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="Model for Heart Disease Analysis",
    page_icon="👋",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

variable_information = """
                        ## Variables in the dataset
                        1. age
                        2. sex
                        3. chest pain type (4 values)
                        4. resting blood pressure
                        5. serum cholestoral in mg/dl
                        6. fasting blood sugar > 120 mg/dl
                        7. resting electrocardiographic results (values 0,1,2)
                        8. maximum heart rate achieved
                        9. exercise induced angina
                        10. oldpeak = ST depression induced by exercise relative to rest
                        11. the slope of the peak exercise ST segment
                        12. number of major vessels (0-3) colored by flourosopy
                        13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
                        """

st.write("# Welcome to Heart Disease data EDA and classification app !")

st.sidebar.write("Click on pages above for details.")

st.markdown(variable_information)

st.markdown(
    """
    ## Necessity 
Instead of building yet another Shiny app, A search for flexible Python deployable ML app where we can quickly look at different 
 model parameters and predict the probability of heart attack (inference tab on model_details page) lead me 
recording following bunch of libraries
 
    Below 📚 are found during 🔬 and not all of them have been researched in much detail (hence feel free to 😒 on any of the comments below)

Plotly dash: albeit Plotly being one of the most common plotting libraries (allowing extensive interactivity) I decided not to go with this option for a few reasons:
I wouldn’t be able to mix plotting libraries if a specific visualization required anything more specific.
we can definitely solve the point above by fiddling with flask and react (used for its backend)
 but this would definitely have me spending more time on it than on the **actual visualization**, to showcase some detaling
 , I have added  Model selection in the dataframe
 
Bokeh: looks like a nice visualization library with some interactivity baked in it. Unfortunately the “dashboarding” part is left to the user.

Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.

After some research I decided to give streamlit a go as it offers the right amount of interactivity and its layout approach, albeit being a bit restrictive, looks flexible enough to churn out something decent in a relatively short amount of time.
"""
)
st.markdown(
    """
    
    
Project structure: 
The project has been structured in the following way:
"""
)
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image('./assets/structure.png', width=150, caption='Project Structure')

with col3:
    st.write(' ')

st.markdown(
    """



This allows to keep the app organized in three distinct blocks with the following logic:

Everything related to plotting and user interaction should live in EDA.py. This is further split into plots and models when building large apps to avoid having too much stuff in the same module
Data preprocessing and helper methods live in different directories.
The dashboard is assembled in Intro.py

"""
)
