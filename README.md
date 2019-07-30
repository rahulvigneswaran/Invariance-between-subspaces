# Invariance Between Subspaces [![Made With python 3.7](https://img.shields.io/badge/Made%20with-Python%203.7-brightgreen)]() [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]() [![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)]() 



## Flowcharts

![FlowChart](readme_assets/baseline.png)

*Fig 1: Flowchart of Invariance Baseline*

------------------------------------------------

![FlowChart](readme_assets/lottery_ticket.png)

*Fig 2: Flowchart of Invariance in Lottery Ticket Hypothesis*

------------------------------------------------









## Requirements
- To install requirements, `pip install -r requirements.txt`.


## To Replicate Results
- For Invariance baseline, run `baseline.py`.
- For Invariance in Lottery Ticket Hypothesis, run `lottery_ticket.py`.
- For Invariance layerwise, run `layerwise.py`

## Repository Structure
```bash
Invariance-between-subspaces
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
├── readme_assets
│   ├── baseline.png
│   ├── layerwise.png
│   └── lottery_ticket.png
└── model
    ├── abstractmodel.py
    ├── forward.py
    ├── fullconn.py
    ├── __init__.py
    └── logreg.py
```


## Issue / Want to Contribute ? :
Open a new issue or do a pull request incase your are facing any difficulty with the code base or you want to contribute to it.

[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://github.com/rahulvigneswaran/Invariance-between-subspaces/issues/new)