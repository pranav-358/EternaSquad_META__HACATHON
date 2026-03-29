"""
invoice_data.py — Synthetic invoice dataset for InvoiceAgentEnv.

Each entry contains:
  - invoice_text : the raw invoice the agent sees
  - ground_truth : the expected extraction / validation / routing answer
  - task_level   : "easy" | "medium" | "hard"

Ground truth keys mirror the InvoiceAction fields the grader checks.
"""

INVOICES = [
    # -----------------------------------------------------------------------
    # EASY — clean, well-formatted invoices. Agent must extract key fields.
    # -----------------------------------------------------------------------
    {
        "task_level": "easy",
        "invoice_text": """
INVOICE

Vendor:       Apex Office Supplies Pvt Ltd
Invoice No:   INV-2024-00891
Invoice Date: 2024-03-15
Due Date:     2024-04-14

Bill To: Scaler Technologies Ltd
         123 MG Road, Bangalore, 560001

Items:
  1. A4 Paper (500 sheets)    Qty: 10   Unit Price: 250.00   Total: 2500.00
  2. Ballpoint Pens (Box/12)  Qty:  5   Unit Price: 120.00   Total:  600.00
  3. Stapler Heavy Duty       Qty:  2   Unit Price: 450.00   Total:  900.00

Subtotal:   4000.00 INR
GST (18%):   720.00 INR
TOTAL:      4720.00 INR

Payment: Bank Transfer
Bank: HDFC Bank  |  A/C: 50100234567890  |  IFSC: HDFC0001234
""",
        "ground_truth": {
            "vendor_name": "Apex Office Supplies Pvt Ltd",
            "invoice_number": "INV-2024-00891",
            "invoice_date": "2024-03-15",
            "due_date": "2024-04-14",
            "subtotal": 4000.00,
            "tax_amount": 720.00,
            "total_amount": 4720.00,
            "currency": "INR",
            "is_valid": True,
            "anomaly_flags": [],
            "routing_department": "finance",
        },
    },
    {
        "task_level": "easy",
        "invoice_text": """
TAX INVOICE

From:   CloudServe Infrastructure Inc.
        Suite 400, 1 Market Street, San Francisco, CA 94105
        Tax ID: US-87-4421009

To:     DataFlow Analytics Corp
        789 Tech Park, Austin, TX 78701

Invoice #:    CI-2024-3301
Date:         2024-01-22
Payment Due:  2024-02-21

Services Rendered:
  Cloud Compute (m5.2xlarge × 720 hrs)   $2,160.00
  Object Storage (5 TB × $0.023/GB)      $  117.76
  Data Transfer (2.1 TB outbound)         $  189.00

Subtotal:    $2,466.76
Sales Tax (8.25%):  $203.51
Amount Due:  $2,670.27 USD
""",
        "ground_truth": {
            "vendor_name": "CloudServe Infrastructure Inc.",
            "invoice_number": "CI-2024-3301",
            "invoice_date": "2024-01-22",
            "due_date": "2024-02-21",
            "subtotal": 2466.76,
            "tax_amount": 203.51,
            "total_amount": 2670.27,
            "currency": "USD",
            "is_valid": True,
            "anomaly_flags": [],
            "routing_department": "engineering",
        },
    },

    # -----------------------------------------------------------------------
    # MEDIUM — invoices with math errors, date issues, or missing fields.
    # Agent must validate and flag problems.
    # -----------------------------------------------------------------------
    {
        "task_level": "medium",
        "invoice_text": """
INVOICE

Vendor:         LegalEdge Consulting LLP
Invoice No.:    LE-2024-0047
Invoice Date:   2024-02-10
Due Date:       2024-01-25          <-- NOTE: due date BEFORE invoice date

Legal Services — January 2024:
  Contract review (8 hrs @ $350/hr)     $2,800.00
  Regulatory compliance advisory (4 hrs) $1,200.00
  Document drafting (6 hrs @ $350/hr)   $2,100.00

Subtotal:    $6,100.00
Tax (0%):       $0.00
Total Due:   $6,500.00              <-- NOTE: total does not match subtotal + tax

Payment Terms: Net 15
""",
        "ground_truth": {
            "vendor_name": "LegalEdge Consulting LLP",
            "invoice_number": "LE-2024-0047",
            "invoice_date": "2024-02-10",
            "due_date": "2024-01-25",
            "subtotal": 6100.00,
            "tax_amount": 0.00,
            "total_amount": 6500.00,
            "currency": "USD",
            "is_valid": False,
            "validation_notes_keywords": ["due date", "before", "total", "mismatch"],
            "anomaly_flags": [],
            "routing_department": "legal",
        },
    },
    {
        "task_level": "medium",
        "invoice_text": """
PROFORMA INVOICE

Supplier:   TechParts Global Pvt. Ltd.
            Plot 77, MIDC Industrial Estate, Pune 411019
GSTIN:      27AABCT1332L1ZV

Buyer:      AutoManufacture India Ltd.

Invoice No:   TPG-2024-1189
Date:         2024-04-05
Due:          2024-05-05

Item Description          HSN    Qty    Rate (INR)   Amount (INR)
----------------------------------------------------------------------
CNC Machined Brackets     7318    50     1,200.00    60,000.00
Precision Bolts M12 (box)  7318   200       85.00    17,000.00
Aluminium Plates 6mm      7606    30     2,500.00    75,000.00
----------------------------------------------------------------------
Subtotal:                                           152,500.00
CGST @ 9%:                                          12,825.00
SGST @ 9%:                                          12,825.00
Total:                                              178,150.00   <-- math error: should be 178,150

Note: Qty for Precision Bolts listed as 200 but PO-2024-0332 requested only 100.
""",
        "ground_truth": {
            "vendor_name": "TechParts Global Pvt. Ltd.",
            "invoice_number": "TPG-2024-1189",
            "invoice_date": "2024-04-05",
            "due_date": "2024-05-05",
            "subtotal": 152500.00,
            "tax_amount": 25650.00,
            "total_amount": 178150.00,
            "currency": "INR",
            "is_valid": False,
            "validation_notes_keywords": ["quantity", "mismatch", "PO", "100"],
            "anomaly_flags": [],
            "routing_department": "finance",
        },
    },

    # -----------------------------------------------------------------------
    # HARD — fraud signals, unusual patterns, complex routing decisions.
    # Agent must detect anomalies AND route correctly.
    # -----------------------------------------------------------------------
    {
        "task_level": "hard",
        "invoice_text": """
INVOICE

From:   Pinnacle IT Solutions
        Registered: 2024-11-01   (Company incorporated only 4 months ago)
        No GST registration number provided
        Contact: pinnacle.it.solutions.2024@gmail.com   (free email domain)

To:     Enterprise Payments Dept
        Scaler Technologies Ltd

Invoice No:    PIS-001           (very low sequential number)
Invoice Date:  2024-03-14
Due Date:      2024-03-16        (only 2-day payment window — unusually urgent)

Services:
  "Digital Transformation Consulting"   $48,500.00
  (No breakdown of hours, deliverables, or personnel provided)

Subtotal:   $48,500.00
Tax:             $0.00
TOTAL:      $48,500.00 USD

Wire to: Cayman Islands account
         Bank: First Caribbean International Bank
         SWIFT: FCIBKYKY
         Account: KY12-3456-7890-1234

Note: Please process urgently — deadline is tomorrow.
""",
        "ground_truth": {
            "vendor_name": "Pinnacle IT Solutions",
            "invoice_number": "PIS-001",
            "invoice_date": "2024-03-14",
            "due_date": "2024-03-16",
            "subtotal": 48500.00,
            "tax_amount": 0.00,
            "total_amount": 48500.00,
            "currency": "USD",
            "is_valid": False,
            "anomaly_flags_required": [
                "new_vendor",         # company very recently incorporated
                "no_tax_id",          # missing GST/tax registration
                "free_email_domain",  # gmail address for large invoice
                "low_invoice_number", # PIS-001 suggests new/shell company
                "urgency_pressure",   # "process urgently", 2-day window
                "offshore_bank",      # Cayman Islands wire
                "vague_description",  # no deliverable breakdown
            ],
            "anomaly_flags_minimum": 3,   # must catch at least 3 of the above
            "routing_department": "legal",
        },
    },
    {
        "task_level": "hard",
        "invoice_text": """
INVOICE

Vendor:     MediSupply Chain Ltd
            GSTIN: 29AACCM9876K1ZP
Invoice No: MSC-2024-7734
Date:       2024-03-01
Due:        2024-03-31

Bill To:    Healthtech Startup India

Items:
  PPE Kits (N95 + Gloves + Gown)   Qty: 1000   Rate: 350   Total: 350,000.00
  Surgical Masks Type IIR           Qty: 5000   Rate:  18   Total:  90,000.00
  Hand Sanitiser 500ml              Qty:  500   Rate: 120   Total:  60,000.00
  Cold Chain Logistics (2–8°C)      Qty:   1    Rate: 25000 Total:  25,000.00

Subtotal:              525,000.00 INR
GST (12% on medical):   63,000.00 INR   <-- 12% of 525,000 = 63,000 ✓
TOTAL:                 588,000.00 INR

Note: Previous invoice MSC-2024-7701 for identical items (₹5,88,000) was submitted
      on 2024-02-28 and is currently pending payment.

Authorized signatory: Dr. Ramesh Verma
""",
        "ground_truth": {
            "vendor_name": "MediSupply Chain Ltd",
            "invoice_number": "MSC-2024-7734",
            "invoice_date": "2024-03-01",
            "due_date": "2024-03-31",
            "subtotal": 525000.00,
            "tax_amount": 63000.00,
            "total_amount": 588000.00,
            "currency": "INR",
            "is_valid": False,
            "anomaly_flags_required": [
                "duplicate_invoice",  # identical to MSC-2024-7701
                "duplicate_amount",
            ],
            "anomaly_flags_minimum": 1,
            "routing_department": "finance",
        },
    },
]