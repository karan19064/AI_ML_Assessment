from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 20)
        self.cell(0, 15, 'AI/ML Assessment Project: Easy Explanation', border=0, new_x="LMARGIN", new_y="NEXT", align='C')
        self.set_font('Helvetica', 'I', 12)
        self.cell(0, 10, 'A simple guide to explain your project in interviews', border=0, new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align='L')
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def chapter_body(self, text):
        self.set_font('Helvetica', '', 12)
        self.multi_cell(0, 8, text)
        self.ln(10)

def create_pdf():
    pdf = PDF()
    pdf.add_page()
    
    pdf.chapter_title("Overview of the Project")
    text_overview = (
        "This project is a complete, production-ready AI and Data Engineering system. "
        "It is divided into three main problem statements. If anyone asks what you built, you can say: "
        "'I built a 3-part system that includes Machine Learning for predictive maintenance, "
        "a high-performance streaming API for log analysis, and an AI Chatbot (RAG) for HR policies.'"
    )
    pdf.chapter_body(text_overview)
    
    pdf.chapter_title("Part 1: Machine Learning (Predictive Maintenance)")
    text_p1 = (
        "The Goal: Predict when a machine or sensor is going to fail.\n\n"
        "What we did:\n"
        "- The data was highly imbalanced (only 0.5% of the data were actual failures). This is common in real life.\n"
        "- We built a robust pipeline using Scikit-Learn that scales the data and handles extreme outliers.\n"
        "- We trained two models: SMOTE + XGBoost, and a Cost-Sensitive Random Forest.\n"
        "- We focused on Business Cost, not just accuracy. Since a False Negative (missing a failure) costs $100 and a False Positive (false alarm) costs $1, we mathematically tuned the probability threshold to minimize the financial cost for the company. XGBoost won, saving the business thousands of dollars."
    )
    pdf.chapter_body(text_p1)
    
    pdf.chapter_title("Part 2: Log Analyzer API (Core Python & Memory Safety)")
    text_p2 = (
        "The Goal: Process massive log files (like 50GB) without crashing the server.\n\n"
        "What we did:\n"
        "- The old code read the entire file into memory at once, causing Out-Of-Memory (OOM) crashes.\n"
        "- We rewrote it using Python Generators (yield) and Context Managers. This means it streams the file line-by-line, keeping the memory footprint at almost zero (O(1) space complexity).\n"
        "- We wrapped this inside a FastAPI backend so users can upload massive files over an API, and the server processes it seamlessly to flag suspicious transactions over $10,000."
    )
    pdf.chapter_body(text_p2)

    pdf.chapter_title("Part 3: Advanced AI Systems (HR Policy RAG)")
    text_p3 = (
        "The Goal: Build an AI assistant that answers employee questions based strictly on company HR policies, without hallucinating.\n\n"
        "What we did:\n"
        "- RAG stands for Retrieval-Augmented Generation. Instead of relying on ChatGPT's generic knowledge, we force the AI to read our specific documents.\n"
        "- We chunked a mock HR policy document and converted the text into embeddings (mathematical vectors) using a local HuggingFace model.\n"
        "- We stored these vectors in ChromaDB (a vector database).\n"
        "- When a user asks 'How much maternity leave do I get?', the system searches ChromaDB for the most relevant paragraphs, and sends ONLY those paragraphs to the LLM to generate a factual answer.\n"
        "- We built a beautiful dark-mode web dashboard to make it easy for non-technical users to interact with both the Log Analyzer and the AI Chatbot."
    )
    pdf.chapter_body(text_p3)
    
    pdf.output('AI_ML_Project_Easy_Explanation.pdf')

if __name__ == '__main__':
    create_pdf()
