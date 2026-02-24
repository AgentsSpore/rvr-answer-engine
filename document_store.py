"""Document store with sample medical research documents"""

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentStore:
    """Manages document corpus and embeddings"""
    
    def __init__(self):
        self.documents: List[Dict[str, str]] = []
        self.embeddings: np.ndarray = None
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def load_sample_documents(self):
        """Load sample medical research documents for demo"""
        # Sample documents covering various diabetes treatments
        sample_docs = [
            {
                'id': 'doc1',
                'title': 'Metformin as First-Line Therapy for Type 2 Diabetes',
                'text': 'Metformin is recommended as the first-line pharmacological treatment for type 2 diabetes. It works by reducing hepatic glucose production and improving insulin sensitivity. Studies show it reduces HbA1c by 1-2% and has cardiovascular benefits.'
            },
            {
                'id': 'doc2',
                'title': 'SGLT2 Inhibitors in Diabetes Management',
                'text': 'SGLT2 inhibitors like empagliflozin and canagliflozin represent a newer class of diabetes medications. They work by blocking glucose reabsorption in the kidneys, promoting urinary glucose excretion. Evidence suggests cardiovascular and renal protective effects.'
            },
            {
                'id': 'doc3',
                'title': 'GLP-1 Receptor Agonists for Glycemic Control',
                'text': 'GLP-1 receptor agonists such as semaglutide and liraglutide enhance insulin secretion and suppress glucagon. Clinical trials demonstrate significant HbA1c reduction and weight loss. These agents show promise for cardiovascular risk reduction.'
            },
            {
                'id': 'doc4',
                'title': 'Lifestyle Modification in Type 2 Diabetes',
                'text': 'Diet and exercise remain cornerstone interventions for type 2 diabetes management. Mediterranean diet, caloric restriction, and 150 minutes of moderate aerobic activity per week can improve glycemic control and reduce medication needs.'
            },
            {
                'id': 'doc5',
                'title': 'Insulin Therapy for Advanced Diabetes',
                'text': 'Insulin therapy becomes necessary when oral agents fail to achieve glycemic targets. Basal-bolus regimens combining long-acting and rapid-acting insulin can effectively manage blood glucose. Continuous glucose monitoring aids in dose optimization.'
            },
            {
                'id': 'doc6',
                'title': 'DPP-4 Inhibitors as Add-On Therapy',
                'text': 'DPP-4 inhibitors like sitagliptin and linagliptin are well-tolerated oral agents that enhance incretin activity. They provide modest HbA1c reduction (0.5-0.8%) with low hypoglycemia risk and are often used in combination therapy.'
            },
            {
                'id': 'doc7',
                'title': 'Bariatric Surgery for Diabetes Remission',
                'text': 'Bariatric surgery, particularly Roux-en-Y gastric bypass and sleeve gastrectomy, can induce diabetes remission in obese patients. Mechanisms include weight loss, altered gut hormone secretion, and improved insulin sensitivity.'
            },
            {
                'id': 'doc8',
                'title': 'Thiazolidinediones for Insulin Resistance',
                'text': 'Thiazolidinediones (TZDs) like pioglitazone improve insulin sensitivity by activating PPAR-gamma receptors. While effective for glycemic control, concerns about weight gain, fluid retention, and cardiovascular effects limit their use.'
            },
            {
                'id': 'doc9',
                'title': 'Continuous Glucose Monitoring Systems',
                'text': 'CGM devices provide real-time glucose data, helping patients and clinicians optimize treatment. Studies show CGM use reduces HbA1c and hypoglycemia risk, particularly when combined with insulin pump therapy.'
            },
            {
                'id': 'doc10',
                'title': 'Alpha-Glucosidase Inhibitors for Postprandial Control',
                'text': 'Acarbose and miglitol delay carbohydrate absorption in the small intestine, reducing postprandial glucose spikes. These agents have modest efficacy and may cause gastrointestinal side effects but are safe with low hypoglycemia risk.'
            },
            {
                'id': 'doc11',
                'title': 'Combination Therapy Strategies',
                'text': 'Combining medications with complementary mechanisms (e.g., metformin + GLP-1 agonist + SGLT2 inhibitor) can achieve better glycemic control than monotherapy. Personalized approaches based on patient characteristics optimize outcomes.'
            },
            {
                'id': 'doc12',
                'title': 'Emerging: Dual GIP/GLP-1 Receptor Agonists',
                'text': 'Tirzepatide, a dual GIP/GLP-1 receptor agonist, demonstrates superior HbA1c reduction and weight loss compared to GLP-1 agonists alone. This represents a promising advancement in diabetes pharmacotherapy.'
            },
            {
                'id': 'doc13',
                'title': 'Pancreatic Islet Transplantation',
                'text': 'Islet cell transplantation is an experimental approach for type 1 and select type 2 diabetes cases. While it can restore insulin production, challenges include donor shortage, immunosuppression requirements, and limited long-term success.'
            },
            {
                'id': 'doc14',
                'title': 'Sulfonylureas in Diabetes Treatment',
                'text': 'Sulfonylureas like glipizide and glyburide stimulate pancreatic insulin secretion. Though effective and inexpensive, they carry hypoglycemia risk and potential for beta-cell exhaustion over time, making them less preferred than newer agents.'
            },
            {
                'id': 'doc15',
                'title': 'Nutritional Supplements and Diabetes',
                'text': 'Some studies suggest chromium, magnesium, and vitamin D supplementation may improve glycemic control, though evidence is mixed. Supplements should complement, not replace, standard medical therapy.'
            }
        ]
        
        self.documents = sample_docs
        self._compute_embeddings()
    
    def _compute_embeddings(self):
        """Pre-compute embeddings for all documents"""
        texts = [doc['text'] for doc in self.documents]
        self.embeddings = self.encoder.encode(texts, show_progress_bar=False)
    
    def add_documents(self, new_docs: List[Dict[str, str]]):
        """Add new documents to the store"""
        self.documents.extend(new_docs)
        self._compute_embeddings()
