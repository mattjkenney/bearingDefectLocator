import streamlit as st

def contact_form_button():

    with st.expander("Contact me"):
        with st.form("Contact Me"):
            st.text_input("Name")
            st.text_input("Email")
            st.text_area("Message")
            st.form_submit_button("Send")