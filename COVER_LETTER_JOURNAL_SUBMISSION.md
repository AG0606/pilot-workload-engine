# Cover Letter for Journal Submission

**To:**  
The Editor-in-Chief  
*IEEE Transactions on Human-Machine Systems* / *IEEE Transactions on Aerospace and Electronic Systems*  

**Date:** October 11, 2026  

**Subject:** Submission of Original Research Manuscript titled *"e-Pilot: A High-Throughput Database-Centric and Deep Multimodal Architecture for Real-Time Pilot Cognitive Workload Assessment and Cockpit Display Decluttering"*

Dear Editor-in-Chief,

We are pleased to submit our original research manuscript titled **"e-Pilot: A High-Throughput Database-Centric and Deep Multimodal Architecture for Real-Time Pilot Cognitive Workload Assessment and Cockpit Display Decluttering"** for consideration as a Regular Paper in *IEEE Transactions on Human-Machine Systems*.

### Background & Research Motivation
Loss of Control In-Flight (LOC-I) is the leading fatal accident category in modern aviation. Investigative analyses by the FAA and NTSB reveal that cognitive failure modes—specifically attentional tunnelling (channelized attention), spatial distraction (diverted attention), and startle responses—are primary precursors to catastrophic pilot error during abnormal flight regimes. While modern glass cockpits provide high volumes of flight and systems data, visual clutter on Primary Flight Displays (PFD) and Engine Indicating and Crew Alerting Systems (EICAS) compounds mental workload during emergencies.

### Key Technical Contributions
In this work, we present an end-to-end software platform and database-centric architecture engineered to assess pilot cognitive states in real time and automate cockpit display decluttering within hard avionics latency constraints ($< 50.0\text{ ms}$):
1. **BCNF Relational Database Architecture:** An 11-entity relational schema normalized to Boyce-Codd Normal Form (BCNF), mathematically eliminating functional dependency redundancies and insertion/deletion anomalies across flight crew, airframe metadata, sensor devices, and high-frequency streams.
2. **Sub-Millisecond ACID Transactional Alert Protocol:** An atomic transaction coordinator that commits safety-critical alerts, display decluttering state updates, and immutable audit logs within $p_{50} = 0.1174\text{ ms}$, with zero deadlocks under concurrent multi-threaded IoT worker streams.
3. **Multi-Rate Synchronization & Boundary Integrity:** Resampling heterogeneous neurological (EEG @ 256 Hz), autonomic (ECG/GSR @ 256 Hz), ocular (gaze @ 60 Hz), and kinematic (telemetry @ 10 Hz) streams onto a unified 20 Hz reference time grid, accompanied by strict gap and session boundary guards.
4. **Deep Cross-Modal Multi-Head Attention Network:** An architecture uniting spatial-temporal EEGNet, cardio 1D residual networks, and multi-head self-attention ($H=4$), optimized via multi-class focal loss ($\gamma=2.0$) with automated inverse class weighting to overcome the severe 80% nominal cruise class imbalance.
5. **Multi-Agent Decision Engine & Display Taxonomy:** A 3-tier display decluttering taxonomy (Level 0 Full, Level 1 Moderate, Level 2 Essentials "Basic Six") regulated by dual-threshold temporal hysteresis to prevent display flickering.
6. **Empirical Verification:** Validated on the NASA / Kaggle commercial aviation benchmark (120,000 empirical samples), achieving 81.91% balanced accuracy, an inference latency of $p_{50} = 0.354\text{ ms}$ ($> 20\times$ faster than the $50\text{ ms}$ avionics budget), and sustained database ingestion throughput of 176,972 records/second.

### Declarations
- This manuscript represents original work that has not been published previously, is not under consideration for publication elsewhere, and will not be submitted elsewhere while under review.
- All authors have approved the manuscript and agree with its submission.
- All code, schemas, and benchmark verification suites are fully open and reproducible.

### Suggested Reviewers
We respectfully suggest the following independent expert reviewers in neuroergonomics, avionics HMI, and real-time database systems:
1. **Prof. Frédéric Dehais**, ISAE-SUPAERO, Neuroergonomics and Human Factors, France (`frederic.dehais@isae-supaero.fr`) — *Expert in flight-deck EEG, attentional tunnelling, and cockpit neuroergonomics.*
2. **Prof. Raphaëlle N. Roy**, ISAE-SUPAERO, Department of Aerospace Vehicles Design and Control, France (`raphaelle.roy@isae.fr`) — *Pioneer in flight simulator EEG mental workload datasets and passive brain-computer interfaces.*
3. **Dr. Chad Weiss**, U.S. Air Force Research Laboratory (AFRL) / MIT Lincoln Laboratory, USA (`chad.weiss@ll.mit.edu`) — *Lead on CogPilot AI challenges and multimodal pilot biometric modeling.*
4. **Prof. Mica R. Endsley**, SA Technologies / Former Chief Scientist of the U.S. Air Force, USA (`mica@satechnologies.com`) — *World authority on situation awareness, human-machine interface design, and cockpit automation.*

Thank you very much for your time and consideration of our manuscript.

Sincerely,

**The Authors**  
Department of Computer Science and Engineering  
Vellore Institute of Technology (VIT), Vellore 632014, India  
Email: `ashutosh.g@vit.ac.in`
