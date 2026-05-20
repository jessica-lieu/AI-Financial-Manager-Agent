# AI-Financial-Manager-Agent

## Details

Manual data entry for expense tracking is inherently inefficient and prone to human error. When budgeting, it is vital to know exactly what you are spending and how that money is being allocated across different categories. However, while receipts log every item and cent, it becomes increasingly difficult to remember those specifics if you are manually filling out a tracker weeks later. This can lead to "blind spots" in your budget, where you may be overspending in certain areas without realizing it. Additionally, manual data entry is time-consuming and tedious, which can make it difficult to stick to a consistent budgeting routine.

## Team

Jessica Lieu (018326048)

## Dataset

- ReceiptQA: A massive dataset of 171,000 QA pairs from 3,500 receipts across Retail, Food, Supermarkets, Fashion, and Medical domains.
- Personal receipts with blurry text, missing fields, and inconsistent formatting

## Approach

The program is a closed-loop feedback cycle. First, the AI will identify key zones of the receipt such as header, line items, and footer. Then those items and numbers will be extracted into a structured format. To ensure accuracy, function calling would be used so that arithmetic would be handled by a Python function for deterministic calculation. Then we would compare the sum against the printed receipt total. If a discrepancy persists after a re-scan, then it would require human intervention.

# How to Run
```
pip install -r requirements.txt
streamlit run main.py
```