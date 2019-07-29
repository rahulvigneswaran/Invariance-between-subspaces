# Invariance Between Subspaces




![FlowChart](Subspace_Invariance_Flowchart.png)









## Requirements
- To install requirements, `pip install -r requirements.txt`.


## To Replicate Results
- For Invariance baseline, run `baseline.py`.
- For Invariance in Lottery Ticket Hypothesis, run `lottery_ticket.py`.
- For Invariance layerwise, run `layerwise.py`

## Repository Structure
```bash
Invariance_between_subspaces
├── baseline.py
├── lottery_ticket.py
├── layerwise.py
├── data.py
├── README.md
├── requirements.txt
├── results
│   ├── baseline
│   ├── layerwise
│   └── lottery_ticket
└── model
    ├── abstractmodel.py
    ├── forward.py
    ├── fullconn.py
    ├── __init__.py
    └── logreg.py
```