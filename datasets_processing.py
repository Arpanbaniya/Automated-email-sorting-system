import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle

def load_data(spam_folder, ham_folder, promo_folder, work_folder):

    spam_emails = []
    spam_labels = []

    try:
      
        for filename in os.listdir(spam_folder):
            if filename.endswith('.txt'):  
                with open(os.path.join(spam_folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
                    spam_emails.append(text)
                    spam_labels.append('spam')
        print(f"Loaded {len(spam_emails)} spam emails from {len(os.listdir(spam_folder))} files.")
    except Exception as e:
        print(f"Error loading spam emails from folder: {e}")


    ham_emails = []
    ham_labels = []

    try:
        for filename in os.listdir(ham_folder):
            if filename.endswith('.txt'):
                with open(os.path.join(ham_folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
                    text = file.read()
                   
                    if 'work' in filename.lower():
                        ham_emails.append(text)
                        ham_labels.append('work')
                    else:
                        ham_emails.append(text)
                        ham_labels.append('personal')
        print(f"Loaded {len(ham_emails)} ham emails from text files.")
    except Exception as e:
        print(f"Error loading text files from ham folder: {e}")

 
    promo_emails = []
    promo_labels = []

    try:
        with open(os.path.join(promo_folder, 'promo_emails.txt'), 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            email_list = content.split('\n---\n') 
            for email in email_list:
                promo_emails.append(email.strip())  
                promo_labels.append('promo')
        print(f"Loaded {len(promo_emails)} promotional emails from promo_emails.txt.")
    except Exception as e:
        print(f"Error loading promotional emails: {e}")

   
    work_emails = []
    work_labels = []

    try:
        with open(os.path.join(work_folder, 'work.txt'), 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            email_list = content.split('\n---\n')
            for email in email_list:
                work_emails.append(email.strip()) 
                work_labels.append('work')
        print(f"Loaded {len(work_emails)} work emails from work.txt.")
    except Exception as e:
        print(f"Error loading work emails: {e}")

 
    all_emails = spam_emails + ham_emails + promo_emails + work_emails
    all_labels = ['spam'] * len(spam_emails) + ham_labels + promo_labels + work_labels

   
    all_labels = [0 if label == 'spam' else 1 if label == 'personal' else 2 if label == 'work' else 3 for label in all_labels]

    return all_emails, all_labels


def prepare_data():
    spam_folder = 'datasets/spam'
    ham_folder = 'datasets/ham'
    promo_folder = 'datasets/promo'
    work_folder = 'datasets/work'
    
    
    emails, labels = load_data(spam_folder, ham_folder, promo_folder, work_folder)


    emails = [str(email) for email in emails]
    
    if len(emails) == 0:
        print("No emails loaded. Please check dataset paths and file types.")
        return None, None, None, None
    

    X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)
    

    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
  
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"Data processing complete. Training data shape: {X_train_vec.shape}, Test data shape: {X_test_vec.shape}")
    
    return X_train_vec, X_test_vec, y_train, y_test

def train_model():
    X_train_vec, X_test_vec, y_train, y_test = prepare_data()

    if X_train_vec is None:
        return

   
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

  
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred))

   
    with open('email_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved as 'email_model.pkl'.")

if __name__ == "__main__":
    train_model()
