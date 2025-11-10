# doc_sum_pipeline.py
import os
import numpy as np
import shutil
from dotenv import load_dotenv
from openai import OpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings

load_dotenv()

embedding_key = os.getenv("EMBEDDING_KEY")
embedding_endpoint = os.getenv("EMBEDDING_ENDPOINT")

chat_endpoint = os.getenv("CHAT_ENDPOINT")
chat_key = os.getenv("CHAT_KEY")


class OpenAIEmbeddingsWrapper(Embeddings):
   
    def __init__(self, client, model_name="text-embedding-ada-002"):
        self.client = client
        self.model_name = model_name

    def embed_documents(self, texts):
     
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                input=text, model=self.model_name
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        """
        Returns a single embedding for a query string.
        """
        response = self.client.embeddings.create(
            input=text, model=self.model_name
        )
        return response.data[0].embedding


class Doc_Summarizer:
    def __init__(self):
        # Clients
        self.embedding_client = OpenAI(api_key=embedding_key, base_url=embedding_endpoint)
        self.embeddings_wrapper = OpenAIEmbeddingsWrapper(self.embedding_client)
        self.chat_client = ChatCompletionsClient(
            endpoint=chat_endpoint,
            credential=AzureKeyCredential(chat_key),
            model="Llama-4-Scout-17B-16E-Instruct"
        )

        # Paths
        self.emd_path = "index_store"
        os.makedirs(self.emd_path, exist_ok=True)

   
    def load_and_chunk_pdf(self, pdf_path, chunk_size=1000, chunk_overlap=200, separators=None):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        if separators is None:
            separators = ["\n\n", "\n", ".", " "]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

        chunked_docs = text_splitter.split_documents(docs)
        return chunked_docs, docs

  
    def create_faiss_index_in_batches(self, all_docs, batch_size=100):
        index_store = self.emd_path

        # Remove any previous FAISS index
        if os.path.exists(os.path.join(index_store, "index.faiss")):
            shutil.rmtree(index_store)
            os.makedirs(index_store, exist_ok=True)

        for i in range(0, len(all_docs), batch_size):
            batch_docs = all_docs[i:i + batch_size]
            print(f"📦 Processing batch {i // batch_size + 1} ({len(batch_docs)} docs)")

            faiss_db = FAISS.from_documents(documents=batch_docs, embedding=self.embeddings_wrapper)

            if os.path.exists(os.path.join(index_store, "index.faiss")):
                local_db = FAISS.load_local(index_store, self.embeddings_wrapper, allow_dangerous_deserialization=True)
                local_db.merge_from(faiss_db)
                local_db.save_local(index_store)
            else:
                faiss_db.save_local(folder_path=index_store)

        print("✅ FAISS index created and merged successfully.")

   
    def retrieve_page(self, query, k=4):
        if not os.path.exists(os.path.join(self.emd_path, "index.faiss")):
            raise FileNotFoundError("FAISS index not found. Please create it first.")

        db = FAISS.load_local(self.emd_path, self.embeddings_wrapper, allow_dangerous_deserialization=True)
        results = db.similarity_search(query, k=k)
        return [i.metadata["page"] for i in results]

   
    def create_summary(self, data):
        prompt = f"""
        You are a financial analyst. Summarize the following text focusing **only** on financial data, metrics, trends, and implications.
        Exclude non-financial content.

        Text:
        {data}
        """

        completion = self.chat_client.complete(
            messages=[
                SystemMessage(content="You are a financial analyst summarizing financial data only."),
                UserMessage(content=prompt),
            ]
        )

        return completion.choices[0].message.content

    def get_summary_from_list_of_pages(self, list_of_pages, docs):
        total_text = "\n\n".join([docs[p].page_content for p in list_of_pages])
        summary = self.create_summary(total_text)
        return summary

    def create_final_summary(self, summaries_text):
        prompt = f"""
        You are a senior financial analyst combining multiple section summaries into a cohesive report.
        Combine all financial information without losing detail or duplicating content.

        Input summaries:
        {summaries_text}
        """

        completion = self.chat_client.complete(
            messages=[
                SystemMessage(content="You are a senior financial analyst combining multiple summaries."),
                UserMessage(content=prompt),
            ]
        )

        return completion.choices[0].message.content

   
    # MAIN WORKFLOW
  
    def summarize_document_workflow(self, topic_json):
        pdf_path = "data/Sample-Financial-Statements-1-pages-2.pdf"
        print(f"📄 Loading {pdf_path}...")

        all_docs, docs = self.load_and_chunk_pdf(pdf_path)
        print(f"📚 Loaded {len(all_docs)} text chunks")

        print("⚙️ Creating FAISS index in batches...")
        self.create_faiss_index_in_batches(all_docs, batch_size=100)

        total_pages = len(docs)
        first_twenty = min(total_pages, 20)
        first_text = "\n\n".join([docs[i].page_content for i in range(first_twenty)])
        first_summary = self.create_summary(first_text)

        topic_wise_summary = {"First Twenty Pages": first_summary}
        relevant_pages = {}

        print("🔍 Retrieving pages for each topic...")
        for topic, details in topic_json.items():
            all_pages = []
            for q in details["retrieval_prompts"]:
                pages = self.retrieve_page(q)
                all_pages.extend(pages)
            relevant_pages[topic] = sorted(set(all_pages))

        print("🧠 Generating topic-wise summaries...")
        for topic, pages in relevant_pages.items():
            if not pages:
                continue
            summary = self.get_summary_from_list_of_pages(pages, docs)
            topic_wise_summary[topic] = summary

        # Combine all summaries
        combined_text = "\n----------\n".join(topic_wise_summary.values())
        final_summary = self.create_final_summary(combined_text)

        os.makedirs("data/doc_summary_outputs", exist_ok=True)
        output_path = os.path.join(
            "data/doc_summary_outputs",
            os.path.basename(pdf_path).replace(".pdf", "_summary.txt")
        )
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_summary)

        print(f"✅ Final summary saved to: {output_path}")
        return final_summary




if __name__ == "__main__":
    
    
    
    
    topic_json = {
  "Company Overview": {
    "questions": [
      "What is the company’s name and primary line of business?",
      "What are the company’s main products or services?",
      "In which markets or regions does the company operate?",
      "What are the company’s subsidiaries or business segments?"
    ],
    "retrieval_prompts": [
      "About the company",
      "Company profile or background",
      "Business overview",
      "Nature of operations or core business activities",
      "Overview of products or services",
      "Corporate structure or subsidiaries"
    ]
  },

  "Financial Highlights": {
    "questions": [
      "What are the key financial highlights for the reporting period?",
      "What is the percentage growth or decline in key metrics like revenue and net profit?",
      "What are the major financial achievements or setbacks compared to the previous year?"
    ],
    "retrieval_prompts": [
      "Financial highlights",
      "Financial summary",
      "Performance overview",
      "Key financial indicators",
      "Annual or quarterly highlights",
      "Summary of results"
    ]
  },

  "Revenue": {
    "questions": [
      "What is the company’s total revenue during the reporting period?",
      "How does the revenue growth rate compare to the previous period?",
      "What are the major sources of revenue?"
    ],
    "retrieval_prompts": [
      "Revenue",
      "Total income",
      "Sales or turnover",
      "Top-line growth",
      "Revenue breakdown",
      "Income statement revenue section"
    ]
  },

  "Expenses": {
    "questions": [
      "What are the company’s total expenses for the reporting period?",
      "What is the amount of Cost of Goods Sold (COGS)?",
      "What are the operating expenses (OPEX)?",
      "What is the depreciation and amortization expense?",
      "What is the interest expense?"
    ],
    "retrieval_prompts": [
      "Expenses",
      "Operating costs",
      "Administrative expenses",
      "COGS or cost of sales",
      "Depreciation and amortization",
      "Interest or finance costs"
    ]
  },

  "Profitability": {
    "questions": [
      "What is the company’s gross profit and gross profit margin?",
      "What is the EBITDA for the period?",
      "What is the EBIT (Operating Profit)?",
      "What is the profit before tax (PBT)?",
      "What is the net profit or profit after tax (PAT)?",
      "What is the net profit margin percentage?"
    ],
    "retrieval_prompts": [
      "Profitability",
      "Income statement bottom line",
      "EBITDA or EBIT",
      "Net profit or PAT",
      "Earnings summary",
      "Profit margin"
    ]
  },

  "Balance Sheet": {
    "questions": [
      "What is the company’s total assets?",
      "What are the current assets and non-current assets?",
      "What are the total liabilities?",
      "What are the current liabilities and non-current liabilities?",
      "What is the shareholders’ equity or net worth?",
      "What is the working capital?"
    ],
    "retrieval_prompts": [
      "Balance sheet",
      "Assets and liabilities",
      "Net worth or equity",
      "Financial position",
      "Working capital details",
      "Statement of financial position"
    ]
  },

  "Cash Flow": {
    "questions": [
      "What is the cash flow from operating activities (CFO)?",
      "What is the cash flow from investing activities (CFI)?",
      "What is the cash flow from financing activities (CFF)?",
      "What is the free cash flow (FCF)?"
    ],
    "retrieval_prompts": [
      "Cash flow statement",
      "Operating cash flow",
      "Investing and financing cash flow",
      "Free cash flow",
      "Net cash generated",
      "Cash inflows and outflows"
    ]
  },

  "Key Financial Ratios": {
    "questions": [
      "What is the current ratio?",
      "What is the quick ratio?",
      "What is the debt-to-equity ratio?",
      "What is the interest coverage ratio?",
      "What is the return on equity (ROE)?",
      "What is the return on assets (ROA)?",
      "What is the return on investment (ROI)?",
      "What is the asset turnover ratio?"
    ],
    "retrieval_prompts": [
      "Financial ratios",
      "Liquidity ratios",
      "Profitability ratios",
      "Leverage ratios",
      "Efficiency ratios",
      "Key performance indicators",
      "KPI analysis"
    ]
  },

  "Shareholder Information": {
    "questions": [
      "What is the earnings per share (EPS)?",
      "What is the dividend per share?",
      "What is the dividend payout ratio?",
      "What is the book value per share?",
      "What is the market capitalization?"
    ],
    "retrieval_prompts": [
      "Shareholder information",
      "Earnings per share (EPS)",
      "Dividend details",
      "Equity and share capital",
      "Book value",
      "Market value or capitalization"
    ]
  },

  "Management Discussion & Analysis": {
    "questions": [
      "What are the management’s key comments on performance?",
      "What risks and opportunities have been identified by management?",
      "What is the future outlook or growth strategy discussed by management?"
    ],
    "retrieval_prompts": [
      "Management discussion",
      "MD&A",
      "Performance commentary",
      "Risk and outlook",
      "Management analysis",
      "Business strategy and outlook"
    ]
  },

  "Auditor’s Report": {
    "questions": [
      "What is the auditor’s opinion on the financial statements?",
      "Are there any qualifications or adverse remarks mentioned?",
      "Does the auditor confirm compliance with accounting standards?"
    ],
    "retrieval_prompts": [
      "Auditor’s report",
      "Independent auditor’s opinion",
      "Audit observations",
      "Compliance with accounting standards",
      "Audit report summary"
    ]
  },

  "Notes to Financial Statements": {
    "questions": [
      "What accounting policies have been followed?",
      "What are the details of contingent liabilities?",
      "What related party transactions are mentioned?",
      "What commitments or pending obligations are reported?"
    ],
    "retrieval_prompts": [
      "Notes to accounts",
      "Explanatory notes",
      "Accounting policies",
      "Contingent liabilities and commitments",
      "Related party disclosures",
      "Supplementary notes"
    ]
  },

  "Corporate Governance": {
    "questions": [
      "What are the corporate governance practices followed by the company?",
      "Who are the members of the Board of Directors?",
      "What are the company’s CSR initiatives?",
      "What risk management practices are mentioned?"
    ],
    "retrieval_prompts": [
      "Corporate governance report",
      "Board of directors",
      "CSR activities",
      "Governance framework",
      "Risk management policies",
      "Leadership and ethics"
    ]
  },

  "Significant Events / Future Outlook": {
    "questions": [
      "What major events or announcements occurred during the reporting period?",
      "What new investments, acquisitions, or divestitures are reported?",
      "What is the company’s future business plan or growth forecast?",
      "What macroeconomic or regulatory factors may affect the company?"
    ],
    "retrieval_prompts": [
      "Significant developments",
      "Future outlook",
      "Strategic initiatives",
      "Major events or acquisitions",
      "Forward-looking statements",
      "Business plan or projections"
    ]
  }
}
    
      # wherever your JSON lives
    
    
    summarizer = Doc_Summarizer()
    summarizer.summarize_document_workflow(topic_json)
