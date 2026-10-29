"""
Step 6: Deployment with Streamlit + Live Gmail / Android App Auto-Filter
Interactive web application for Email/SMS Spam Detection with real-time
NLP preprocessing inspection, TF-IDF keyword attribution, multi-model comparison,
and live Gmail IMAP synchronization that labels spam in the Gmail Android app.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from gmail_watcher import load_processed_cache, scan_gmail_inbox
from preprocessing import preprocess_steps_breakdown, preprocess_text
from train_model import train_and_evaluate

load_dotenv()

ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "models", "spam_artifacts.joblib")
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")

SAMPLE_EMAILS = {
    "Select an example email...": "",
    "🚨 Spam: Prize / Gift Card Winner": (
        "URGENT! Congratulations! Your email address has been selected as the winner of a "
        "$1,000 Amazon Gift Card and a FREE luxury holiday package! Click http://claim-prize-now.biz "
        "within 24 hours to claim your reward or call 0800-999-888 immediately!"
    ),
    "🚨 Spam: Phishing Account Alert": (
        "FINAL SECURITY NOTICE: We detected unauthorized login attempts on your bank account. "
        "Your account access has been temporarily suspended. Please verify your identity immediately "
        "at http://secure-bank-login-update.com to avoid permanent closure and $250 penalty fees."
    ),
    "✅ Ham: Work Meeting & Project Update": (
        "Hi team, please find attached the Q3 product roadmap and meeting notes from yesterday's sync. "
        "Let's meet tomorrow at 2:30 PM in Conference Room B to finalize the deployment timeline. "
        "Let me know if you need any adjustments."
    ),
    "✅ Ham: Personal Lunch Catch-up": (
        "Hey Alex! Are we still on for coffee and lunch this Friday around 12:30? "
        "I finished reviewing the presentation slides you sent over earlier—they look great. See you soon!"
    ),
}


@st.cache_resource
def load_artifacts():
    """Load trained models and TF-IDF vectorizer, training automatically if needed."""
    if not os.path.exists(ARTIFACTS_PATH):
        with st.spinner("Training spam detection models for the first time..."):
            return train_and_evaluate()
    return joblib.load(ARTIFACTS_PATH)


def get_top_contributing_tokens(vectorizer, text_clean: str, top_n: int = 8) -> pd.DataFrame:
    """Return the highest-scoring TF-IDF tokens present in the input email."""
    tfidf_vec = vectorizer.transform([text_clean])
    nonzero_indices = tfidf_vec.nonzero()[1]
    if len(nonzero_indices) == 0:
        return pd.DataFrame(columns=["Token / N-gram", "TF-IDF Score"])

    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = tfidf_vec.data
    order = np.argsort(scores)[::-1][:top_n]

    return pd.DataFrame(
        {
            "Token / N-gram": feature_names[nonzero_indices[order]],
            "TF-IDF Score": np.round(scores[order], 4),
        }
    )


def save_credentials_to_env(email_addr: str, app_pw: str, label_name: str) -> None:
    """Save Gmail settings to the local .env file (excluded from git via .gitignore)."""
    content = (
        f"GMAIL_ADDRESS={email_addr.strip()}\n"
        f"GMAIL_APP_PASSWORD={app_pw.strip()}\n"
        f"GMAIL_SPAM_LABEL={label_name.strip()}\n"
    )
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    st.set_page_config(
        page_title="Email Spam Detection System",
        page_icon="📧",
        layout="wide",
    )

    st.title("📧 Email & SMS Spam Detection System")
    st.markdown(
        "**End-to-End NLP & Machine Learning Pipeline:** "
        "`Data Collection` ➔ `Text Preprocessing` ➔ `TF-IDF Vectorization` ➔ "
        "`Model Training` ➔ `Evaluation` ➔ `Gmail & Android Sync`"
    )

    artifacts = load_artifacts()
    vectorizer = artifacts["vectorizer"]
    models = artifacts["models"]
    best_model_name = artifacts["best_model_name"]
    metrics = artifacts["metrics"]
    stats = artifacts["dataset_stats"]

    # Sidebar: Model Selection & Dataset Summary
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        model_names = list(models.keys())
        default_idx = model_names.index(best_model_name) if best_model_name in model_names else 0
        selected_model_name = st.selectbox(
            "Select Classification Algorithm",
            options=model_names,
            index=default_idx,
            help="Choose between Multinomial Naive Bayes, Logistic Regression, and Support Vector Machine (SVM).",
        )
        if selected_model_name == best_model_name:
            st.caption("⭐ **Recommended:** Highest F1-Score on test set")

        st.divider()
        st.subheader("📊 Selected Model Metrics")
        m = metrics[selected_model_name]
        c1, c2 = st.columns(2)
        c1.metric("Precision", f"{m['precision']:.2%}")
        c2.metric("Recall", f"{m['recall']:.2%}")
        c3, c4 = st.columns(2)
        c3.metric("F1-Score", f"{m['f1_score']:.2%}")
        c4.metric("Accuracy", f"{m['accuracy']:.2%}")

        st.divider()
        st.subheader("🗂️ Dataset Statistics")
        st.write(f"- **Total Messages:** {stats['total_messages']:,}")
        st.write(f"- **Legitimate (Ham):** {stats['ham_count']:,}")
        st.write(f"- **Spam Messages:** {stats['spam_count']:,}")
        st.write(f"- **TF-IDF Vocabulary:** {stats['vocab_size']:,} features")

    tab_predict, tab_gmail, tab_metrics = st.tabs(
        [
            "🔍 Live Email Classifier",
            "📲 Gmail & Android Auto-Filter",
            "📈 Model Evaluation & Comparison",
        ]
    )

    with tab_predict:
        sample_choice = st.selectbox("Try a sample email or paste your own below:", list(SAMPLE_EMAILS.keys()))
        default_text = SAMPLE_EMAILS[sample_choice]

        email_input = st.text_area(
            "Email / Message Body",
            value=default_text,
            height=170,
            placeholder="Paste an email subject and body here to classify it as Spam or Ham...",
        )

        col_btn, _ = st.columns([1, 4])
        analyze_clicked = col_btn.button("🚀 Analyze Email", type="primary", use_container_width=True)

        if analyze_clicked or default_text:
            if not email_input.strip():
                st.warning("Please enter or select an email message to classify.")
            else:
                steps = preprocess_steps_breakdown(email_input)
                clean_text = steps["5_stemmed_final"]
                X_vec = vectorizer.transform([clean_text])

                model = models[selected_model_name]
                prediction = int(model.predict(X_vec)[0])
                proba = model.predict_proba(X_vec)[0]
                spam_prob = float(proba[1])

                st.subheader("Prediction Result")
                res_col1, res_col2 = st.columns([2, 1])

                with res_col1:
                    if prediction == 1:
                        st.error("### 🚨 SPAM DETECTED\nThis message exhibits strong spam or phishing characteristics.")
                    else:
                        st.success("### ✅ HAM (LEGITIMATE EMAIL)\nThis message appears safe and legitimate.")

                with res_col2:
                    st.metric(
                        label="Spam Probability",
                        value=f"{spam_prob:.1%}",
                        delta="High Risk" if prediction == 1 else "Low Risk",
                        delta_color="inverse",
                    )
                    st.progress(spam_prob)

                # Side-by-side predictions across all 3 models
                with st.expander("🤖 Consensus Across All Trained Models", expanded=True):
                    consensus_rows = []
                    for m_name, m_obj in models.items():
                        pred_i = int(m_obj.predict(X_vec)[0])
                        prob_i = float(m_obj.predict_proba(X_vec)[0][1])
                        consensus_rows.append(
                            {
                                "Model": m_name,
                                "Prediction": "🚨 SPAM" if pred_i == 1 else "✅ HAM",
                                "Spam Confidence": f"{prob_i:.1%}",
                                "Ham Confidence": f"{1.0 - prob_i:.1%}",
                            }
                        )
                    st.dataframe(pd.DataFrame(consensus_rows), use_container_width=True, hide_index=True)

                # Inspect Step 2 & Step 3
                col_nlp, col_tfidf = st.columns(2)
                with col_nlp:
                    with st.expander("🧼 Step 2: NLP Preprocessing Breakdown", expanded=False):
                        st.markdown("**1. Lowercased & Cleaned (No URLs/Punctuation):**")
                        st.code(steps["2_cleaned_no_punct"] or "(empty)", language="text")
                        st.markdown(f"**2. Tokens ({len(steps['3_tokens'])}):**")
                        st.write(steps["3_tokens"][:30])
                        st.markdown(f"**3. After Stop Word Removal ({len(steps['4_without_stopwords'])}):**")
                        st.write(steps["4_without_stopwords"][:30])
                        st.markdown("**4. Final Stemmed Text fed to TF-IDF:**")
                        st.code(clean_text or "(empty)", language="text")

                with col_tfidf:
                    with st.expander("📐 Step 3: Top TF-IDF Features in Message", expanded=False):
                        top_tokens_df = get_top_contributing_tokens(vectorizer, clean_text)
                        if top_tokens_df.empty:
                            st.info("No known vocabulary tokens found in the message.")
                        else:
                            st.dataframe(top_tokens_df, use_container_width=True, hide_index=True)

    with tab_gmail:
        st.subheader("📲 Automatic Gmail & Android Gmail App Integration")
        st.markdown(
            "Connect your Gmail account to automatically scan incoming emails and attach the **`AI-Spam`** label "
            "(while keeping them safely in your Inbox). Because Gmail labels sync across the cloud, "
            "**every flagged email immediately displays an `AI-Spam` badge inside your Android Gmail app!**"
        )

        with st.expander("🔑 How to get a 16-character Google App Password (1-minute setup)", expanded=False):
            st.markdown(
                """
                1. Go to your [Google Account Security Settings](https://myaccount.google.com/security) and ensure **2-Step Verification** is turned **ON**.
                2. Open [Google App Passwords](https://myaccount.google.com/apppasswords).
                3. Enter an app name like **`Email Spam Detector`** and click **Create**.
                4. Copy the **16-character password** shown on screen and paste it below (or save it to `.env`).
                5. In your **Android Gmail App**, open the left sidebar menu ➔ scroll to **Labels** ➔ tap **`AI-Spam`** to review all flagged emails in one tap, or see the **`AI-Spam`** tag directly next to emails in your Inbox!
                """
            )

        c_form1, c_form2, c_form3 = st.columns([2, 2, 1])
        with c_form1:
            gmail_addr = st.text_input(
                "Gmail Address",
                value=os.getenv("GMAIL_ADDRESS", ""),
                placeholder="your_email@gmail.com",
            )
        with c_form2:
            gmail_app_pw = st.text_input(
                "16-Character Google App Password",
                value=os.getenv("GMAIL_APP_PASSWORD", ""),
                type="password",
                placeholder="abcd efgh ijkl mnop",
            )
        with c_form3:
            custom_label = st.text_input(
                "Gmail Label Name",
                value=os.getenv("GMAIL_SPAM_LABEL", "AI-Spam"),
            )

        opt_col1, opt_col2, opt_col3, opt_col4 = st.columns([1.2, 1.2, 1.5, 1.5])
        with opt_col1:
            max_to_scan = st.number_input("Recent Emails to Scan", min_value=1, max_value=100, value=15)
        with opt_col2:
            unread_only = st.checkbox("Unread emails only", value=False)
        with opt_col3:
            force_rescan = st.checkbox("Re-scan already checked emails", value=False)
        with opt_col4:
            move_to_spam = st.checkbox(
                "Move directly to Spam folder",
                value=False,
                help="Unchecked (Recommended): Keeps email safely in Inbox and attaches the 'AI-Spam' label.",
            )

        btn_col1, btn_col2, _ = st.columns([1.5, 1.5, 2])
        scan_now = btn_col1.button("🔄 Scan Gmail Inbox Now", type="primary", use_container_width=True)
        save_env = btn_col2.button("💾 Save Settings to .env", use_container_width=True)

        if save_env:
            if not gmail_addr or not gmail_app_pw:
                st.warning("Please enter both your Gmail address and App Password before saving.")
            else:
                save_credentials_to_env(gmail_addr, gmail_app_pw, custom_label)
                st.success("Saved credentials to `.env`! You can now also run `python gmail_watcher.py` in the background.")

        if scan_now:
            with st.spinner(f"Connecting to {gmail_addr} via secure IMAP and classifying recent messages..."):
                results, err = scan_gmail_inbox(
                    email_address=gmail_addr,
                    app_password=gmail_app_pw,
                    label_name=custom_label,
                    max_emails=int(max_to_scan),
                    unread_only=unread_only,
                    move_to_spam=move_to_spam,
                    model_name=selected_model_name,
                    force_rescan=force_rescan,
                )
            if err:
                st.error(err)
            elif not results:
                st.info("No matching emails found in your Inbox.")
            else:
                spam_count = sum(1 for r in results if r["prediction"] == "SPAM")
                st.success(
                    f"Scanned **{len(results)}** emails using **{selected_model_name}** — "
                    f"flagged **{spam_count}** message(s) with label **`{custom_label}`**!"
                )
                df_res = pd.DataFrame(results)[
                    ["prediction", "spam_probability", "from", "subject", "action_taken", "date"]
                ].rename(
                    columns={
                        "prediction": "Status",
                        "spam_probability": "Spam Confidence",
                        "from": "From",
                        "subject": "Subject",
                        "action_taken": "Gmail / Android Action",
                        "date": "Date",
                    }
                )
                df_res["Status"] = df_res["Status"].map(
                    lambda x: "🚨 SPAM" if x == "SPAM" else "✅ HAM"
                )
                df_res["Spam Confidence"] = df_res["Spam Confidence"].map(lambda p: f"{p:.1%}")
                st.dataframe(df_res, use_container_width=True, hide_index=True)

        # Show previously scanned history if available
        cached_history = load_processed_cache()
        if cached_history and not scan_now:
            with st.expander(f"📜 Previously Scanned Emails ({len(cached_history)} cached)", expanded=False):
                hist_list = list(reversed(list(cached_history.values())))
                st.dataframe(pd.DataFrame(hist_list), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### ⚙️ Run 24/7 Background Auto-Filter Daemon")
        st.markdown(
            "To continuously watch your Gmail account in the background (even when the browser is closed) "
            "and automatically label new emails for your Android Gmail app every 60 seconds, run:"
        )
        st.code("python gmail_watcher.py --interval 60", language="bash")

    with tab_metrics:
        st.subheader("📈 Step 5: Model Evaluation Metrics Comparison")
        st.markdown(
            "Because email datasets are typically imbalanced (~87% Ham, ~13% Spam), "
            "**Precision**, **Recall**, and **F1-Score** are tracked alongside Accuracy:"
        )
        comparison_rows = []
        for m_name, m_vals in metrics.items():
            comparison_rows.append(
                {
                    "Model": m_name + (" ⭐" if m_name == best_model_name else ""),
                    "Precision": m_vals["precision"],
                    "Recall": m_vals["recall"],
                    "F1-Score": m_vals["f1_score"],
                    "Accuracy": m_vals["accuracy"],
                }
            )
        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(
            comp_df.style.format(
                {
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-Score": "{:.2%}",
                    "Accuracy": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(f"Confusion Matrix — {selected_model_name}")
        cm = metrics[selected_model_name]["confusion_matrix"]
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Ham (0)", "Actual: Spam (1)"],
            columns=["Predicted: Ham (0)", "Predicted: Spam (1)"],
        )
        st.table(cm_df)


if __name__ == "__main__":
    main()
