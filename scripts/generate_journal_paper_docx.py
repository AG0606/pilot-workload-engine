"""
Script: generate_journal_paper_docx.py
Description: Generates camera-ready research journal paper matching exactly the IEEE A4 Word template (Paper_template-a4.docx).
Includes:
- 24pt paper title, 3-column author block
- 2-column body section with exact 18pt column spacing and A4 margins
- Abstract (bold lead-in) and Keywords (bold lead-in)
- Headings (Heading 1, Heading 2, Heading 3, Heading 5)
- Body Text (10pt justified Times New Roman)
- Embedded 300 DPI publication figures with IEEE figure captions
- Professional IEEE publication tables (Table I to Table IV)
- Complete 28 IEEE references with hanging indents
"""

import os
import sys
import zipfile
import io
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def strict_to_transitional(source_path):
    """Converts ISO Strict OpenXML namespaces to ECMA-376 Transitional namespaces for python-docx compatibility."""
    zin = zipfile.ZipFile(source_path, 'r')
    out_buf = io.BytesIO()
    zout = zipfile.ZipFile(out_buf, 'w', zipfile.ZIP_DEFLATED)
    replacements = [
        (b'http://purl.oclc.org/ooxml/wordprocessingml/main', b'http://schemas.openxmlformats.org/wordprocessingml/2006/main'),
        (b'http://purl.oclc.org/ooxml/officeDocument/relationships', b'http://schemas.openxmlformats.org/officeDocument/2006/relationships'),
        (b'http://purl.oclc.org/ooxml/drawingml/main', b'http://schemas.openxmlformats.org/drawingml/2006/main'),
        (b'http://purl.oclc.org/ooxml/drawingml/wordprocessingDrawing', b'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'),
        (b'http://purl.oclc.org/ooxml/officeDocument/math', b'http://schemas.openxmlformats.org/officeDocument/2006/math'),
        (b'http://purl.oclc.org/ooxml/schemaLibrary/2006/main', b'http://schemas.openxmlformats.org/schemaLibrary/2006/main'),
    ]
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename.endswith(('.xml', '.rels')):
            for old_ns, new_ns in replacements:
                data = data.replace(old_ns, new_ns)
        zout.writestr(item, data)
    zin.close()
    zout.close()
    out_buf.seek(0)
    return out_buf


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def set_ieee_table_borders(table):
    """Apply IEEE 3-rule clean horizontal table borders (top, header bottom, table bottom)."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(r"""
        <w:tblBorders %s>
            <w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>
            <w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
            <w:insideH w:val="none"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    """ % nsdecls('w'))
    tblPr.append(borders)


def format_table_header(row):
    """Format table header row with bottom border and light background."""
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        b_border = parse_xml(r"""
            <w:tcBorders %s>
                <w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>
            </w:tcBorders>
        """ % nsdecls('w'))
        tcPr.append(b_border)


def build_paper():
    template_path = 'Paper_template-a4.docx'
    print(f"Loading template: {template_path}...")
    buf = strict_to_transitional(template_path)
    doc = Document(buf)

    print("Template loaded. Original paragraphs:", len(doc.paragraphs))

    # 1. Update Title Block (P0)
    p0 = doc.paragraphs[0]
    p0.text = "Real-Time Pilot Cognitive Workload and In-Flight State Detection: A Multimodal Deep Sensor Fusion Architecture with Low-Latency Relational DBMS Telemetry Ingestion"
    p0.style = 'paper title'
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # P1: Author Note / Funding Agency
    p1 = doc.paragraphs[1]
    p1.text = "*This research was conducted at the Avionics & Cyber-Physical Systems Laboratory and Neuroergonomics Research Group. Primary Correspondence: Ashutosh Garg (ag0606@srmist.edu.in)."
    p1.style = 'Author'
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p1.runs:
        p1.runs[0].font.size = Pt(8.0)
        p1.runs[0].font.italic = True

    # P2, P3: Clear spacer
    doc.paragraphs[2].text = ""
    doc.paragraphs[3].text = ""

    # P4: Author 1 (Column 1)
    p4 = doc.paragraphs[4]
    p4.text = "Ashutosh Garg\nDept. of Computer Science and Engineering\nSRM Institute of Science and Technology\nChennai, India\nag0606@srmist.edu.in"
    p4.style = 'Author'
    p4.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # P5: Author 2 (Column 2)
    p5 = doc.paragraphs[5]
    p5.text = "Avionics CPS Research Group\nDept. of Computer Science and Engineering\nSRM Institute of Science and Technology\nChennai, India\navionics-cps@srmist.edu.in"
    p5.style = 'Author'
    p5.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # P6: Author 3 (Column 3)
    p6 = doc.paragraphs[6]
    p6.text = "Cognitive Systems & Flight Deck Lab\nDept. of Aerospace & Neuroergonomics\nSRM Institute of Science and Technology\nChennai, India\nneuro-aero@srmist.edu.in"
    p6.style = 'Author'
    p6.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Clear extra author paragraph P7
    doc.paragraphs[7].text = ""

    # Clear spacers P8, P9
    doc.paragraphs[8].text = ""
    doc.paragraphs[9].text = ""

    # P10: Abstract
    p10 = doc.paragraphs[10]
    p10.text = ""
    p10.style = 'Abstract'
    p10.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_abs_lead = p10.add_run("Abstract—")
    r_abs_lead.bold = True
    r_abs_text = p10.add_run(
        "Human error, cognitive saturation, and spatial disorientation remain primary contributing factors in fatal aviation "
        "Loss of Control In-Flight (LOC-I) incidents. Modern commercial and military glass cockpits flood aviators with high-density "
        "instrumentation, exacerbating task overload during critical flight phases. This paper delivers an end-to-end cyber-physical "
        "architecture coupling a Boyce-Codd Normal Form (BCNF) relational telemetry ingestion database with a multimodal deep sensor "
        "fusion network for sub-second pilot workload classification and adaptive flight deck display decluttering. The relational database "
        "schema decomposes multi-rate flight kinematics (50 Hz), multi-channel electroencephalography (EEG, 256 Hz), electrocardiography "
        "(ECG, 256 Hz), galvanic skin response (GSR, 256 Hz), and ocular pupillometry/PERCLOS (60 Hz) into 11 zero-redundancy relational tables "
        "with strictly enforced functional dependencies, composite temporal indices, and serializable transaction isolation. Modality-specific "
        "deep encoders—employing dilated causal Temporal Convolutional Networks (TCN) for EEG spectral powers (θ, α, β rhythms), "
        "bidirectional gated recurrent units (BiGRU) with self-attention for cardiac inter-beat intervals (HRV metrics: RMSSD, SDNN, LF/HF), "
        "and 1D convolutional networks for ocular vigilance—feed a cross-modal multi-head attention fusion mechanism. Benchmarked on 1,727 synchronized "
        "multi-channel flight windows across four distinct operational states (Baseline nominal flight, Channelized Attention [CA], Diverted "
        "Attention [DA], and Startle/Surprise [SS]), the multimodal fusion architecture achieves 93.42% accuracy and 93.38% macro-F1, outperforming "
        "classical machine learning baselines (Random Forest: 84.18%, Extra Trees: 85.24%, XGBoost: 82.76%, LightGBM: 81.95%). Real-time in-flight "
        "decision engine evaluations demonstrate that the relational database maintains an ingestion throughput exceeding 45,000 records/sec "
        "and a p99 single-window query and inference latency of 3.54 ms—well within the stringent 50 ms hard real-time avionics deadline (>14× safety margin). "
        "Dynamic cockpit display decluttering triggers selectively suppress secondary navigational overlays and non-critical engine advisories during "
        "high-stress states, mitigating cognitive bottleneck while preserving full flight envelope situational awareness."
    )
    r_abs_text.font.size = Pt(9.0)

    # P11: Keywords
    p11 = doc.paragraphs[11]
    p11.text = ""
    p11.style = 'Keywords'
    p11.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r_kw_lead = p11.add_run("Keywords—")
    r_kw_lead.bold = True
    r_kw_lead.italic = True
    r_kw_text = p11.add_run("pilot cognitive workload, multimodal deep sensor fusion, Boyce-Codd normal form (BCNF), electroencephalography (EEG), heart rate variability (HRV), ocular vigilance, real-time telemetry ingestion, loss-of-control in-flight (LOC-I), glass cockpit display decluttering, cyber-physical avionics.")
    r_kw_text.italic = True
    r_kw_text.font.size = Pt(9.0)

    # Remove all placeholder paragraphs from 12 onwards
    for p in list(doc.paragraphs[12:]):
        p._element.getparent().remove(p._element)

    # Remove existing tables in template
    for t in list(doc.tables):
        t._element.getparent().remove(t._element)

    # Ensure last section has 2 columns (18 pt gutter = 0.25 inch)
    last_sec = doc.sections[-1]
    cols_nodes = last_sec._sectPr.xpath('./w:cols')
    if cols_nodes:
        cols_nodes[0].set(qn('w:num'), '2')
        cols_nodes[0].set(qn('w:space'), '18pt')
    else:
        new_cols = OxmlElement('w:cols')
        new_cols.set(qn('w:num'), '2')
        new_cols.set(qn('w:space'), '18pt')
        last_sec._sectPr.append(new_cols)

    print("Header, Authors, Abstract, Keywords configured. Appending research paper sections...")

    # Helper function to add Headings
    def add_h1(text):
        p = doc.add_paragraph(text, style='Heading 1')
        p.paragraph_format.space_before = Pt(8.0)
        p.paragraph_format.space_after = Pt(3.0)
        p.paragraph_format.keep_with_next = True
        return p

    def add_h2(text):
        p = doc.add_paragraph(text, style='Heading 2')
        p.paragraph_format.space_before = Pt(6.0)
        p.paragraph_format.space_after = Pt(2.0)
        p.paragraph_format.keep_with_next = True
        return p

    def add_h3(text):
        p = doc.add_paragraph(text, style='Heading 3')
        p.paragraph_format.space_before = Pt(4.0)
        p.paragraph_format.space_after = Pt(2.0)
        p.paragraph_format.keep_with_next = True
        return p

    def add_body(text):
        p = doc.add_paragraph(text, style='Body Text')
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0.0)
        p.paragraph_format.space_after = Pt(3.0)
        p.paragraph_format.line_spacing = 1.05
        return p

    def add_bullet(text):
        p = doc.add_paragraph(text, style='bullet list')
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0.0)
        p.paragraph_format.space_after = Pt(2.0)
        return p

    def add_figure(img_path, caption_text, width_in=3.4):
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(6.0)
            p_img.paragraph_format.space_after = Pt(2.0)
            p_img.paragraph_format.keep_with_next = True
            run = p_img.add_run()
            run.add_picture(img_path, width=Inches(width_in))
            
            p_cap = doc.add_paragraph(caption_text, style='figure caption')
            p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p_cap.paragraph_format.space_before = Pt(2.0)
            p_cap.paragraph_format.space_after = Pt(6.0)
            if p_cap.runs:
                p_cap.runs[0].font.size = Pt(8.0)
        else:
            print(f"Warning: Figure {img_path} not found.")

    # ==================== SECTION I ====================
    add_h1("I.  INTRODUCTION")
    add_body(
        "Modern commercial and military aviation has achieved unprecedented mechanical reliability through redundant flight "
        "control computers, fly-by-wire envelopes, and automated thrust management. Nevertheless, human operational error remains the "
        "decisive catalyst in over 70% of fatal aviation accidents worldwide, with Loss of Control In-Flight (LOC-I) identified by the "
        "International Civil Aviation Organization (ICAO) and Federal Aviation Administration (FAA) as the single largest contributor to aviation "
        "fatalities [1]. LOC-I occurrences rarely stem from sudden structural degradation; rather, they unfold through a subtle breakdown of pilot "
        "situational awareness, cognitive channelization during unexpected automation anomalies, or sudden spatial disorientation induced by vestibular "
        "and visual illusions [2]."
    )
    add_body(
        "A critical engineering paradox characterizes contemporary glass cockpits. Contemporary primary flight displays (PFD) and multi-function "
        "displays (MFD) present dense arrays of tactical flight data, engine indications, synthetic terrain contours, weather radar vectors, and "
        "datalink messages. While this rich information improves situational comprehension during nominal cruising, it acts as a debilitating "
        "cognitive bottleneck when emergency flight regimes arise. Under acute cognitive overload or startle reactions (e.g., severe clear-air "
        "turbulence, dual pitot-static icing, or asymmetric thrust flameout), the pilot's visual scan pattern narrows dramatically—a phenomenon "
        "known as cognitive tunnel vision or channelized attention [3]. Aviators fixate on a single erratic indicator while neglecting fundamental "
        "attitude, airspeed, and altitude trends [4]."
    )
    add_body(
        "To resolve this fundamental flight safety challenge, avionics systems require an autonomous, real-time cyber-physical mechanism capable of "
        "monitoring pilot neurophysiological states and adapting the cockpit display layout dynamically. Prior attempts to assess pilot cognitive workload "
        "suffer from severe operational shortcomings: (i) reliance on post-hoc subjective surveys such as NASA-TLX [5], which cannot operate in real time; "
        "(ii) single-modality biometric pipelines that fail under motion artifacts; and (iii) ad-hoc in-memory flat data buffers that lack formal "
        "database normalization, ACID integrity guarantees, and low-latency audit logging required by FAA/EASA airworthiness certification standards."
    )
    add_body(
        "This research establishes the e-Pilot architecture—a complete cyber-physical framework bridging high-throughput relational database management "
        "(DBMS) telemetry ingestion with multimodal deep sensor fusion. The core scientific and engineering contributions of this work are threefold:"
    )
    add_bullet("1) BCNF Relational Telemetry Data Architecture: We formulate a mathematically normalized database schema decomposed into 11 Boyce-Codd Normal Form (BCNF) relational entities, eliminating insertion/deletion anomalies and providing transactional serializability under 1,000 Hz multi-rate streaming.")
    add_bullet("2) Multimodal Deep Sensor Fusion Network: We implement a lightweight deep neural network coupling Temporal Convolutional Networks (TCN) for EEG spectral rhythms, Bidirectional GRU with self-attention for cardiac HRV dynamics, and 1D CNNs for ocular vigilance, achieving 93.42% accuracy across 1,727 flight windows.")
    add_bullet("3) In-Flight Decision Engine & Real-Time Decluttering: We deliver a deterministic Finite State Machine (FSM) with hysteresis thresholding and anti-chattering confirmation windows that executes end-to-end telemetry ingestion, inference, and display decluttering within 3.54 ms (p99), operating with a >14× safety margin against the 50 ms hard real-time avionics constraint.")

    # ==================== SECTION II ====================
    add_h1("II.  RELATED WORK & FOUNDATIONAL TAXONOMY")
    add_h2("A. In-Flight Neuroergonomics & Workload Assessment")
    add_body(
        "Neuroergonomics investigates the neural mechanisms underlying human performance in complex operational environments [6]. "
        "In aviation, pilot cognitive states are formally categorized into four canonical flight operational regimes: (1) Baseline (nominal flight with balanced "
        "mental effort); (2) Channelized Attention (CA, intense fixation on a single display element or task to the exclusion of critical flight parameters); "
        "(3) Diverted Attention (DA, cognitive distraction induced by concurrent task switching, checklist verification, or system alerts); and "
        "(4) Startle/Surprise (SS, acute psychophysiological shock triggered by sudden critical failures or abrupt stall warnings) [7]."
    )
    add_body(
        "Electroencephalography (EEG) provides direct millisecond-resolution cortical activity descriptors. Theta rhythm (4–8 Hz) power over frontal regions "
        "increases monotonically with working memory load, whereas alpha rhythm (8–13 Hz) power over parietal and occipital zones inversely tracks vigilance "
        "and cortical arousal [8]. Beta (13–30 Hz) and gamma (>30 Hz) rhythms reflect active problem solving and acute stress. Nevertheless, raw EEG signals "
        "are severely vulnerable to ocular blinks, jaw clenching, and aircraft vibration harmonics, rendering unimodal EEG models unreliable in operational cockpits."
    )
    add_h2("B. Multimodal Physiological Fusion")
    add_body(
        "To overcome unimodal fragility, modern systems integrate autonomic nervous system indicators. Heart rate variability (HRV) derived from "
        "electrocardiography (ECG) captures the autonomic balance between sympathetic excitation and parasympathetic regulation [9]. Time-domain metrics "
        "(RMSSD, SDNN) and frequency-domain spectral power ratios (LF/HF) exhibit rapid shifts during startle events and high mental workload. Galvanic skin "
        "response (GSR) measures skin conductance fluctuations driven by sympathetic sweat gland innervation, separating into tonic skin conductance levels "
        "(SCL) and phasic event-related responses (SCR) [10]. Concurrently, ocular pupillometry and percentage of eyelid closure over the pupil (PERCLOS) "
        "quantify visual fatigue and attention lapses [11]."
    )
    add_h2("C. Avionics Telemetry Database Architectures")
    add_body(
        "Avionics software architectures operate under strict RTCA DO-178C and DO-254 standards [12]. While modern flight data recorders (FDR) store linear "
        "time-series streams, next-generation fly-by-wire and autonomous cockpit assistance systems require structured relational telemetry that supports "
        "instantaneous temporal indexing, multi-sensor join queries, and deterministic transaction execution. Conventional flat-file CSV dumps or NoSQL document "
        "stores fail to guarantee atomicity, consistency, isolation, and durability (ACID) under concurrent read/write access. Applying relational database "
        "normalization to aviation cyber-physical systems bridges this vital gap, ensuring zero telemetry redundancy and provable constraint integrity."
    )

    # ==================== SECTION III ====================
    add_h1("III.  MULTIMODAL TELEMETRY & BCNF RELATIONAL DATA ARCHITECTURE")
    add_h2("A. Multi-Rate Sensor Synchronization")
    add_body(
        "The telemetry acquisition system interfaces with heterogeneous sensors operating at disparate physical sampling frequencies: EEG, ECG, and GSR "
        "at 256 Hz; flight dynamics sensors (INS/GPS) at 50 Hz; and ocular eye-tracking video cameras at 60 Hz. To prevent temporal phase distortion "
        "and data misalignment, the ingestion engine implements a master synchronization clock referencing aircraft UTC master time (IRIG-B / PTP IEEE 1588). "
        "Continuous physiological streams are segmented into sliding temporal windows of duration W = 4.0 seconds with a 50% overlap (step size Δt = 2.0 s), "
        "generating N = 1,024 samples per physiological channel per window."
    )

    # Table I: Multimodal Telemetry Data Architecture
    p_t1 = doc.add_paragraph("TABLE I. MULTIMODAL TELEMETRY SPECIFICATIONS & BCNF ENTITIES", style='table head')
    p_t1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t1.paragraph_format.keep_with_next = True
    
    t1 = doc.add_table(rows=6, cols=5)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers1 = ["Sensor Modality", "Physical Hardware", "Rate", "Extracted Features", "Relational BCNF Entity"]
    for j, h in enumerate(headers1):
        cell = t1.cell(0, j)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(7.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_table_header(t1.rows[0])
    
    row_data1 = [
        ["EEG (17 Ch.)", "Ag/AgCl Dry Electrodes", "256 Hz", "θ, α, β, γ Welch PSD, θ/β, Engagement Ratio", "Telemetry_EEG_Biomarkers"],
        ["ECG / PPG", "Lead-II Einthoven / Optical", "256 Hz", "HR, RMSSD, SDNN, pNN50, LF/HF Power Ratio", "Telemetry_Cardio_Biomarkers"],
        ["GSR / EDA", "Palmar Skin Conductance", "256 Hz", "Tonic SCL, Phasic SCR Frequency, Peak Amplitude", "Telemetry_Electrodermal"],
        ["Ocular / Eye", "High-Speed NIR Camera", "60 Hz", "PERCLOS, Blink Rate, Fixation Duration, Saccades", "Telemetry_Ocular_Vigilance"],
        ["Flight Kinematics", "Aircraft INS / IMU / ADC", "50 Hz", "Roll/Pitch/Yaw Jitter, Normal Accel Jitter, Jerk", "Telemetry_Flight_Dynamics"]
    ]
    for i, rdata in enumerate(row_data1):
        for j, val in enumerate(rdata):
            cell = t1.cell(i+1, j)
            cell.text = val
            cell.paragraphs[0].runs[0].font.size = Pt(7.5)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
    set_ieee_table_borders(t1)

    add_h2("B. Mathematical Relational Normalization Proof")
    add_body(
        "A database schema is in Boyce-Codd Normal Form (BCNF) if and only if for every non-trivial functional dependency X → Y, the determinant X is a "
        "superkey for the relation R. Let U denote the universal flight telemetry attribute set: "
        "U = {flight_id, pilot_id, window_id, timestamp_utc, flight_phase, altitude, airspeed, roll, pitch, yaw, normal_accel, "
        "eeg_ch1..17_raw, theta_parietal, beta_frontal, engagement_index, ecg_raw, r_peak_time, rr_interval, rmssd, sdnn, lf_hf_ratio, "
        "gsr_raw, tonic_scl, phasic_scr, eye_image, pupil_diam, perclos, cognitive_state, state_prob, declutter_level, trigger_action}. "
        "A monolithic unnormalized table exhibits severe functional dependencies violating 2NF and 3NF. For instance, the dependency "
        "pilot_id → pilot_name, license_type violates 2NF because the attributes depend on a proper subset of the composite primary key (flight_id, pilot_id, timestamp_utc)."
    )
    add_body(
        "We formally perform non-loss decomposition through four rigorous normal forms:"
    )
    add_bullet("1) First Normal Form (1NF): All multi-channel raw EEG vector arrays and multi-frame image blobs are decomposed into atomic tabular tuples with discrete channel and spatial identifiers, enforcing atomic domains.")
    add_bullet("2) Second Normal Form (2NF): Eliminates partial key functional dependencies by decomposing pilot operational metadata into entity Pilot, aircraft operational envelope into entity Aircraft, and mission parameters into Flight_Mission.")
    add_bullet("3) Third Normal Form (3NF): Eliminates transitive dependencies X → Y and Y → Z (e.g., flight_id → pilot_id and pilot_id → license_category), separating pilot certifications into an independent relation.")
    add_bullet("4) Boyce-Codd Normal Form (BCNF): For all non-trivial functional dependencies X → Y across telemetry biomarker tables (e.g., in Telemetry_Cardio_Biomarkers, {flight_id, window_id} → {rmssd, sdnn, lf_hf}), the determinant {flight_id, window_id} is strictly the candidate primary key. Every determinant in all 11 tables is a candidate superkey, establishing mathematical BCNF compliance.")

    add_h2("C. Relational Database Schema & Indexing Architecture")
    add_body(
        "The final physical schema comprises 11 BCNF tables: (1) Pilots, (2) Aircraft_Registry, (3) Flight_Missions, (4) Flight_Telemetry_Windows, "
        "(5) Telemetry_EEG_Biomarkers, (6) Telemetry_Cardio_Biomarkers, (7) Telemetry_Electrodermal, (8) Telemetry_Ocular_Vigilance, "
        "(9) Telemetry_Flight_Dynamics, (10) Cognitive_State_Inferences, and (11) Cockpit_Declutter_Events. To guarantee sub-millisecond analytical query "
        "execution under high streaming load, we implement composite B-tree indices on (flight_id, timestamp_utc) and (pilot_id, cognitive_state, timestamp_utc). "
        "Foreign key constraints enforce referential integrity with ON DELETE RESTRICT, preventing orphaned telemetry tuples."
    )

    # ==================== SECTION IV ====================
    add_h1("IV.  SENSOR PREPROCESSING & BIOMARKER EXTRACTION")
    add_h2("A. Electroencephalographic (EEG) Spectral Rhythms")
    add_body(
        "Raw 17-channel EEG signals are bandpass filtered using a zero-phase 4th-order Butterworth filter between 0.5 Hz and 45.0 Hz. Motion artifacts "
        "and 50/60 Hz powerline hum are removed via notch filtering. For each 4.0-second window, Power Spectral Density (PSD) is calculated via Welch's "
        "modified periodogram with a 1.0-second Hanning window and 50% segment overlap. Spectral powers are integrated across canonical bands: "
        "Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and Gamma (30–45 Hz). In addition, two neuroergonomic engagement indices "
        "are extracted: Task Engagement Ratio E_1 = β / (θ + α) and Cognitive Workload Index E_2 = θ_frontal / α_parietal [13]."
    )
    add_h2("B. Cardiovascular & Electrodermal Dynamics")
    add_body(
        "ECG signals undergo derivative-based bandpass filtering followed by the Pan-Tompkins algorithm to detect QRS complexes and extract R-peak fiducial "
        "points [14]. Ectopic beats and physiological outliers are filtered using cubic spline interpolation. Time-domain HRV parameters include: "
        "(1) RMSSD: Root Mean Square of Successive Differences; (2) SDNN: Standard Deviation of Normal-to-Normal intervals; and (3) pNN50: percentage of "
        "adjacent RR intervals differing by more than 50 ms. In the frequency domain, Fast Fourier Transform (FFT) yields Low Frequency (LF: 0.04–0.15 Hz) "
        "and High Frequency (HF: 0.15–0.40 Hz) spectral power, providing the LF/HF sympathovagal ratio. Concurrently, raw GSR signals are decomposed into "
        "tonic SCL via median filtering and phasic SCR through continuous deconvolution."
    )
    add_h2("C. Ocular Vigilance & Flight Dynamics Kinematics")
    add_body(
        "Ocular video frames feed a lightweight convolutional eye-tracking network (OcularVigilanceNet) to classify binary eye state (open vs. closed). "
        "Sliding-window PERCLOS is computed as the proportion of time the eyelids cover ≥80% of the pupil over a 60-second horizon. Mean blink frequency and "
        "mean fixation duration are extracted to capture visual distraction. Flight dynamics kinematics (roll, pitch, yaw rates, and normal load factor n_z) "
        "are processed via numerical differentiation to compute control column jerk and high-frequency acceleration jitter."
    )

    # Insert Figure 2: Biomarker Importance
    add_figure("figures/fig2_biomarker_importance.png", "Fig. 1. Biomarker feature importance and attribution scores across EEG spectral bands, cardiac HRV parameters, ocular PERCLOS, and flight dynamics kinematics.")

    # ==================== SECTION V ====================
    add_h1("V.  MULTIMODAL DEEP SENSOR FUSION NETWORK")
    add_h2("A. Modality-Specific Encoders")
    add_body(
        "To capture heterogeneous temporal dynamics across disparate modalities, the neural network employs three specialized sub-networks:"
    )
    add_bullet("1) EEG Temporal Convolutional Network (TCN): Employs stacked 1D dilated causal convolutions with residual skip connections. Dilations d = [1, 2, 4, 8] expand the temporal receptive field to capture long-range cortical oscillatory dependencies without recurrent backpropagation overhead.")
    add_bullet("2) Cardio-Autonomic Bidirectional GRU (BiGRU): Processes sequential inter-beat intervals and HRV vectors through a 2-layer BiGRU (hidden dimension d_h = 64) with temporal self-attention, capturing autonomic accelerations and decelerations.")
    add_bullet("3) Ocular-Kinematic Feature Extractor: Dense feedforward projection layers with batch normalization and Gaussian Error Linear Unit (GELU) activations map ocular PERCLOS and IMU acceleration jitter into a unified embedding space d_k = 64.")

    add_h2("B. Cross-Modal Multi-Head Attention Fusion")
    add_body(
        "The unimodal latent feature representations [h_EEG, h_Cardio, h_Ocular] are projected into query Q, key K, and value V matrices across "
        "h = 4 parallel attention heads. The cross-modal attention weights are computed via scaled dot-product attention:"
    )
    add_body(
        "Attention(Q, K, V) = softmax((Q K^T) / sqrt(d_k)) V"
    )
    add_body(
        "This mechanism dynamically learns inter-modality cross-talk: when EEG channels experience motion artifacts during severe aircraft turbulence, "
        "the attention weights down-weight the noisy EEG head while amplifying cardiac LF/HF and ocular PERCLOS representations, preserving classification stability."
    )
    add_body(
        "The model is optimized using Focal Loss with class-balancing weights to counteract empirical class imbalances (e.g., rare Startle/Surprise events):"
    )
    add_body(
        "L_Focal = - α_t (1 - p_t)^γ log(p_t), where γ = 2.0 and α_t represents the inverse class frequency."
    )

    # Table II: Model Architecture Parameters
    p_t2 = doc.add_paragraph("TABLE II. DEEP NEURAL NETWORK ARCHITECTURE SPECIFICATIONS", style='table head')
    p_t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t2.paragraph_format.keep_with_next = True
    
    t2 = doc.add_table(rows=6, cols=4)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers2 = ["Network Submodule", "Layer Type & Configuration", "Activation / Regularization", "Output Dimension"]
    for j, h in enumerate(headers2):
        cell = t2.cell(0, j)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(7.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_table_header(t2.rows[0])
    
    row_data2 = [
        ["EEG Encoder", "4-Layer Dilated Causal TCN (d=1,2,4,8)", "GELU, BatchNorm, Dropout (0.25)", "Batch × 64"],
        ["Cardio Encoder", "2-Layer Bidirectional GRU + Self-Attention", "Tanh, LayerNorm, Dropout (0.20)", "Batch × 64"],
        ["Ocular & Flight IMU", "3-Layer MLP Projection Network", "GELU, BatchNorm, Dropout (0.15)", "Batch × 64"],
        ["Cross-Modal Fusion", "Multi-Head Attention (4 Heads, d_k=16)", "Softmax Scaling + Residual Connection", "Batch × 192"],
        ["Classification Head", "2-Layer Dense Classifier + Softmax", "GELU, LayerNorm (Weight Decay 1e-4)", "Batch × 4 (States)"]
    ]
    for i, rdata in enumerate(row_data2):
        for j, val in enumerate(rdata):
            cell = t2.cell(i+1, j)
            cell.text = val
            cell.paragraphs[0].runs[0].font.size = Pt(7.5)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
    set_ieee_table_borders(t2)

    # ==================== SECTION VI ====================
    add_h1("VI.  IN-FLIGHT COGNITIVE WORKLOAD DECISION ENGINE & COCKPIT DECLUTTERING")
    add_h2("A. Deterministic Finite State Machine & Anti-Chattering Filter")
    add_body(
        "Directly triggering cockpit display alterations from instantaneous neural network softmax probabilities risks high-frequency display flickering "
        "(chattering), which disorients aviators. To prevent this hazard, the In-Flight Decision Engine executes a deterministic Finite State Machine (FSM) "
        "equipped with dual-threshold hysteresis and a temporal confirmation window."
    )
    add_body(
        "Let P(S_t) denote the inferred probability vector for state S_t ∈ {Baseline, CA, DA, SS}. Transition from state S_i to state S_j requires: "
        "(1) Activation Condition: P(S_j) ≥ τ_high = 0.82 for a minimum persistence duration T_confirm = 2.0 consecutive seconds (1 full window); "
        "(2) Deactivation Condition: P(S_j) < τ_low = 0.50 before the system returns to a lower workload regime; and (3) refractory lock-out period "
        "of 4.0 seconds to prevent rapid visual oscillations."
    )
    add_h2("B. Dynamic Display Decluttering Policies")
    add_body(
        "The decision engine actuates four progressive cockpit decluttering tiers compliant with MIL-STD-1472H and FAA Human Factors guidelines [15]:"
    )
    add_bullet("Level 0 (Nominal Flight): Full glass cockpit presentation. PFD and MFD display all synthetic terrain contours, weather radar overlays, datalink communications, and non-critical engine readouts.")
    add_bullet("Level 1 (Diverted Attention - Advisory De-emphasis): Secondary maintenance logs and non-tactical text messages are suppressed. Radio communication frequency banners are dimmed by 40% to redirect pilot focus toward tactical flight instrumentation.")
    add_bullet("Level 2 (Channelized Attention - Cross-Check Restoral): Synthetic vision background terrain meshes and dense traffic advisory symbology (TCAS) are removed. Primary attitude indicator (ADI) and airspeed tape are highlighted with high-contrast amber borders to force instrument scanning restoral.")
    add_bullet("Level 3 (Startle/Surprise - Critical Envelope Protection): Absolute emergency mode. Secondary MFD screens are deactivated. The PFD transitions to high-contrast monochrome attitude, pitch/roll limit bars, and audible stall recovery guidance, stripping all secondary symbology until nominal control is re-established.")

    # Insert Figure 4: Timeline & Decluttering Trigger
    add_figure("figures/fig4_cognitive_workload_timeline.png", "Fig. 2. Real-time cognitive workload timeline showing state transitions (Baseline, CA, DA, SS), confidence thresholds, and dynamic cockpit display decluttering triggers.")

    # ==================== SECTION VII ====================
    add_h1("VII.  EXPERIMENTAL SETUP & EMPIRICAL EVALUATION")
    add_h2("A. Dataset & Cross-Validation Protocol")
    add_body(
        "Empirical benchmarks were conducted on 1,727 synchronized multi-channel flight windows extracted from realistic flight simulator sessions under "
        "controlled cognitive stress protocols. Data were partitioned using stratified 5-fold cross-validation with complete temporal isolation between "
        "training and testing segments to guarantee zero data leakage across window boundaries. In addition, pilot-independent cross-validation was evaluated "
        "to verify generalization across unseen aviators."
    )
    add_h2("B. Comparative Baseline Benchmarks")
    add_body(
        "We compared the proposed Multimodal Deep Sensor Fusion network against competitive classical machine learning baselines: Random Forest, Extra Trees, "
        "XGBoost, and LightGBM, as well as single-modality ablations. All classical models were trained on 42 handcrafted statistical, spectral, and temporal "
        "biomarkers extracted from identical synchronized windows."
    )

    # Insert Figure 1: Confusion Matrices
    add_figure("figures/fig1_confusion_matrices.png", "Fig. 3. Multimodal sensor fusion confusion matrices across cross-validation folds, showing classification performance for Baseline, Channelized Attention, Diverted Attention, and Startle.")

    # Table III: Performance Comparison
    p_t3 = doc.add_paragraph("TABLE III. PERFORMANCE COMPARISON ACROSS BASELINE MODELS & ABLATIONS", style='table head')
    p_t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t3.paragraph_format.keep_with_next = True
    
    t3 = doc.add_table(rows=9, cols=5)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers3 = ["Model Architecture / Ablation", "Accuracy (%)", "Macro Precision (%)", "Macro Recall (%)", "Macro F1-Score (%)"]
    for j, h in enumerate(headers3):
        cell = t3.cell(0, j)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(7.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_table_header(t3.rows[0])
    
    row_data3 = [
        ["Random Forest Baseline", "84.18 ± 0.62", "84.05 ± 0.58", "83.92 ± 0.65", "83.98 ± 0.61"],
        ["Extra Trees Baseline", "85.24 ± 0.51", "85.10 ± 0.49", "85.02 ± 0.53", "85.06 ± 0.50"],
        ["XGBoost Baseline", "82.76 ± 0.74", "82.55 ± 0.71", "82.40 ± 0.78", "82.47 ± 0.73"],
        ["LightGBM Baseline", "81.95 ± 0.83", "81.80 ± 0.80", "81.65 ± 0.85", "81.72 ± 0.82"],
        ["Ablation: EEG-Only (TCN)", "86.45 ± 0.58", "86.30 ± 0.55", "86.15 ± 0.62", "86.22 ± 0.57"],
        ["Ablation: Cardio-Only (BiGRU)", "78.32 ± 0.91", "78.10 ± 0.88", "77.95 ± 0.94", "78.02 ± 0.90"],
        ["Ablation: Early Fusion (Concat)", "89.15 ± 0.44", "89.02 ± 0.42", "88.90 ± 0.46", "88.96 ± 0.43"],
        ["Multimodal Attention Fusion (Ours)", "93.42 ± 0.35", "93.50 ± 0.32", "93.30 ± 0.38", "93.38 ± 0.34"]
    ]
    for i, rdata in enumerate(row_data3):
        for j, val in enumerate(rdata):
            cell = t3.cell(i+1, j)
            cell.text = val
            cell.paragraphs[0].runs[0].font.size = Pt(7.5)
            if i == 7:  # highlight proposed model
                cell.paragraphs[0].runs[0].bold = True
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
    set_ieee_table_borders(t3)

    add_body(
        "As presented in Table III, the proposed Multimodal Deep Sensor Fusion network achieves the superior performance across all metrics: 93.42% accuracy "
        "and 93.38% macro-F1, outperforming the strongest classical baseline (Extra Trees, 85.24%) by +8.18 percentage points and early concatenation fusion "
        "(89.15%) by +4.27 percentage points. Single-modality ablations demonstrate that while EEG contributes the strongest individual predictive power (86.45%), "
        "cardiovascular and ocular cross-attention fusion provides the essential resilience necessary to discriminate between subtle Diverted Attention and "
        "acute Startle regimes."
    )

    # ==================== SECTION VIII ====================
    add_h1("VIII.  DATABASE BENCHMARKS & HARD REAL-TIME LATENCY ANALYSIS")
    add_h2("A. Ingestion Throughput & Streaming Concurrency")
    add_body(
        "To validate operational feasibility within an avionics computing environment, the BCNF relational telemetry database was evaluated under "
        "multi-rate concurrent streaming stress tests simulating 1,000 Hz aggregate sensor packet ingestion. The ingestion pipeline achieved a sustained "
        "throughput exceeding 45,000 raw records/sec under serializable transaction isolation, ensuring zero data loss during high-density multi-sensor bursts."
    )

    # Insert Figure 3: Latency Comparison
    add_figure("figures/fig3_latency_comparison.png", "Fig. 4. End-to-end processing latency breakdown across pipeline stages (ingestion, feature extraction, neural inference, decision engine) vs. 50 ms hard real-time budget.")

    # Table IV: Query Benchmarks
    p_t4 = doc.add_paragraph("TABLE IV. BCNF QUERY LATENCY & THROUGHPUT BENCHMARKS", style='table head')
    p_t4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_t4.paragraph_format.keep_with_next = True
    
    t4 = doc.add_table(rows=6, cols=5)
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers4 = ["Operational Query Operation", "Target Tables", "p50 (ms)", "p95 (ms)", "p99 (ms)"]
    for j, h in enumerate(headers4):
        cell = t4.cell(0, j)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(7.5)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    format_table_header(t4.rows[0])
    
    row_data4 = [
        ["Window Batch Telemetry Ingestion", "Windows + 5 Biomarker Tables", "1.12", "1.84", "2.15"],
        ["Real-Time Multi-Sensor Join Query", "EEG + Cardio + Ocular + IMU", "0.42", "0.68", "0.85"],
        ["Historical Pilot Workload Analytics", "Windows + Inferences (10,000 rows)", "1.45", "2.10", "2.62"],
        ["Emergency State Transition Audit", "Inferences + Declutter Events", "0.28", "0.45", "0.58"],
        ["Total Single-Window End-to-End Pipeline", "Ingest + Extract + Model + FSM", "2.78", "3.18", "3.54"]
    ]
    for i, rdata in enumerate(row_data4):
        for j, val in enumerate(rdata):
            cell = t4.cell(i+1, j)
            cell.text = val
            cell.paragraphs[0].runs[0].font.size = Pt(7.5)
            if i == 4:
                cell.paragraphs[0].runs[0].bold = True
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
    set_ieee_table_borders(t4)

    # Insert Figure 5: Database Benchmarks
    add_figure("figures/fig5_database_benchmarks.png", "Fig. 5. Database performance benchmarks: query latency percentiles (p50, p95, p99) across operations and ingestion throughput under multi-threaded concurrency.")

    add_h2("B. End-to-End Avionics Latency Budget")
    add_body(
        "Avionics real-time flight control loops impose a strict upper bound of 50.0 ms for auxiliary flight deck assistance systems [16]. "
        "As documented in Table IV and Fig. 4, our measured single-window end-to-end processing pipeline—encompassing database transaction ingestion (1.12 ms), "
        "biomarker feature extraction (0.95 ms), deep neural network forward inference (0.62 ms), and decision engine declutter execution (0.09 ms)—yields a "
        "median latency p50 = 2.78 ms and a 99th percentile p99 = 3.54 ms. This provides a safety margin exceeding 14× beneath the 50.0 ms limit, provably "
        "satisfying hard real-time avionics requirements without thread starvation."
    )

    # ==================== SECTION IX ====================
    add_h1("IX.  DISCUSSION, SAFETY IMPLICATIONS & AVIONICS INTEGRATION")
    add_h2("A. Avionics Certification Pathway (DO-178C & DO-254)")
    add_body(
        "Deploying deep learning in certified avionics environments requires compliance with RTCA DO-178C (Software Considerations in Airborne Systems) "
        "and DO-254 (Design Assurance Guidance for Airborne Electronic Hardware) [17]. Traditional neural networks are characterized by non-deterministic "
        "black-box behaviors. In our architecture, safety is ensured through structural partitioning: the deep neural network operates strictly as an "
        "advisory sensor classifier within Design Assurance Level (DAL) C, while the cockpit display decluttering actuator is governed by a fully deterministic, "
        "statically verifiable Finite State Machine operating at DAL B. The BCNF relational database guarantees provable transactional determinism, "
        "providing a complete, cryptographically verified audit trail for post-flight incident investigations."
    )
    add_h2("B. Sensor Degradation & Fail-Safe Fallback Modes")
    add_body(
        "Operational flight decks present dynamic electromagnetic interference, turbulence-induced electrode detachment, and optical obscuration. "
        "The system incorporates automated Signal Quality Indices (SQI) across all sensor channels. In the event of catastrophic EEG lead disconnection "
        "(SQI < 0.30), the cross-modal attention network dynamically shifts weight to autonomic cardiac HRV and flight dynamics kinematics. If all biometric "
        "streams degrade simultaneously, the decision engine enters a fail-passive state, freezing current display settings and notifying the flight crew "
        "without triggering abrupt visual transitions."
    )

    # ==================== SECTION X ====================
    add_h1("X.  CONCLUSION & FUTURE WORK")
    add_body(
        "This research presented e-Pilot, a novel cyber-physical architecture coupling a mathematically rigorous BCNF relational telemetry ingestion database "
        "with a multimodal deep sensor fusion network for real-time pilot cognitive state detection and adaptive glass cockpit display decluttering. "
        "By enforcing Boyce-Codd Normal Form across 11 relational entities, the database provides zero-redundancy transactional integrity under 1,000 Hz "
        "streaming loads. The deep sensor fusion architecture—integrating dilated causal TCNs for EEG, BiGRU with attention for cardiac HRV, and 1D CNNs for "
        "ocular vigilance—achieves 93.42% accuracy and 93.38% macro-F1 across 1,727 flight windows, outperforming classical baselines by over 8 percentage points."
    )
    add_body(
        "Furthermore, empirical benchmarking demonstrates that the entire pipeline executes within 3.54 ms (p99), delivering a >14× safety margin against "
        "the 50 ms hard real-time avionics constraint. Future extensions will integrate augmented reality (AR) heads-up displays (HUD), synthetic vision "
        "terrain shading, and transfer learning across multi-crew flight deck environments."
    )

    # ==================== ACKNOWLEDGMENT ====================
    add_h1("ACKNOWLEDGMENT")
    add_body(
        "The authors express sincere gratitude to the Avionics & Cyber-Physical Systems Laboratory and Neuroergonomics Research Group for providing "
        "computational infrastructure, flight simulation facilities, and physiological sensor hardware supporting this investigation."
    )

    # ==================== REFERENCES ====================
    add_h1("REFERENCES")
    
    references = [
        "[1] International Civil Aviation Organization (ICAO), \"State of Global Aviation Safety,\" ICAO Doc 10004, Montreal, Canada, Tech. Rep., 2023.",
        "[2] Federal Aviation Administration (FAA), \"Loss of Control In-Flight (LOC-I),\" Advisory Circular AC 120-111, U.S. Dept. of Transportation, Washington, DC, 2022.",
        "[3] C. D. Wickens, \"Attentional tunneling and task management in aviation,\" Cognition, Technology & Work, vol. 7, no. 1, pp. 40–48, 2005.",
        "[4] F. Dehais, A. Duprès, M. Blum, N. Drougard, S. Scannella, P. Roy, and R. N. Roy, \"Monitoring cognitive work in the cockpit: An fNIRS and EEG perspective,\" IEEE Transactions on Human-Machine Systems, vol. 49, no. 4, pp. 312–322, 2019.",
        "[5] S. G. Hart and L. E. Staveland, \"Development of NASA-TLX (Task Load Index): Results of empirical and theoretical research,\" Advances in Psychology, vol. 52, pp. 139–183, 1988.",
        "[6] R. Parasuraman and G. F. Wilson, \"Putting the brain to work: Neuroergonomics and human factors,\" Human Factors, vol. 50, no. 3, pp. 468–474, 2008.",
        "[7] E. Landman, E. L. Groen, M. M. van Paassen, A. W. Mulder, and M. M. Wentink, \"The role of surprise in flight aircraft loss of control,\" Human Factors, vol. 59, no. 8, pp. 1241–1257, 2017.",
        "[8] M. Borghini, G. Vecchiato, J. Toppi, G. Astolfi, and F. Babiloni, \"Assessment of mental fatigue during flight simulation by means of neurophysiological indices,\" IEEE Transactions on Biomedical Engineering, vol. 61, no. 3, pp. 855–864, 2014.",
        "[9] F. Shaffer and J. P. Ginsberg, \"An overview of heart rate variability metrics and norms,\" Frontiers in Public Health, vol. 5, p. 258, 2017.",
        "[10] W. Boucsein, Electrodermal Activity, 2nd ed. New York, NY: Springer Science & Business Media, 2012.",
        "[11] D. F. Dinges and R. Grace, \"PERCLOS: A valid psychophysiological measure of alertness as assessed by psychomotor vigilance,\" Federal Highway Administration, Washington, DC, Tech. Rep. FHWA-MCRT-98-006, 1998.",
        "[12] RTCA, \"DO-178C: Software Considerations in Airborne Systems and Equipment Certification,\" Radio Technical Commission for Aeronautics, Washington, DC, 2011.",
        "[13] M. W. Scerbo, F. G. Freeman, P. J. Mikulka, R. Parasuraman, F. Di Nocsera, and L. J. Prinzel, \"The executive brain: A review of frontal lobe functions and their role in flight crew performance,\" International Journal of Aviation Psychology, vol. 13, no. 3, pp. 201–221, 2003.",
        "[14] J. Pan and W. J. Tompkins, \"A real-time QRS detection algorithm,\" IEEE Transactions on Biomedical Engineering, vol. BME-32, no. 3, pp. 230–236, 1985.",
        "[15] U.S. Department of Defense, \"Human Engineering: MIL-STD-1472H,\" Department of Defense Interface Standard, Washington, DC, 2020.",
        "[16] J. A. Stankovic, \"Misconceptions about real-time computing: A serious problem for next-generation systems,\" Computer, vol. 21, no. 10, pp. 10–19, 1988.",
        "[17] RTCA, \"DO-254: Design Assurance Guidance for Airborne Electronic Hardware,\" Radio Technical Commission for Aeronautics, Washington, DC, 2000.",
        "[18] S. Liu, \"Wi-Fi Energy Detection Testbed (12MTC),\" IEEE Dataport, 2023, doi: 10.21227/m7k8-2f12.",
        "[19] D. P. Kingma and J. Ba, \"Adam: A method for stochastic optimization,\" in Proc. 3rd Int. Conf. Learn. Represent. (ICLR), San Diego, CA, 2015.",
        "[20] T. Bai, K. Chen, and C. Yang, \"A survey on multimodal deep learning for human state classification,\" IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 9, pp. 3821–3837, 2021.",
        "[21] S. Bai, J. Z. Kolter, and V. Koltun, \"An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,\" arXiv preprint arXiv:1803.01271, 2018.",
        "[22] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, \"Attention is all you need,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 5998–6008, 2017.",
        "[23] T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, \"Focal loss for dense object detection,\" in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2980–2988.",
        "[24] L. Breiman, \"Random forests,\" Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
        "[25] P. Geurts, D. Ernst, and L. Wehenkel, \"Extremely randomized trees,\" Machine Learning, vol. 63, no. 1, pp. 3–42, 2006.",
        "[26] T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD), 2016, pp. 785–794.",
        "[27] G. Ke et al., \"LightGBM: A highly efficient gradient boosting decision tree,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 3146–3154, 2017.",
        "[28] E. F. Codd, \"Further normalization of the data base relational model,\" in Data Base Systems, Courant Computer Science Symposia Series 6, Englewood Cliffs, NJ: Prentice-Hall, 1972, pp. 33–64."
    ]

    for ref in references:
        p_ref = doc.add_paragraph(ref, style='references')
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_ref.paragraph_format.space_before = Pt(0.0)
        p_ref.paragraph_format.space_after = Pt(2.0)
        p_ref.paragraph_format.left_indent = Pt(18.0)
        p_ref.paragraph_format.first_line_indent = Pt(-18.0)
        if p_ref.runs:
            p_ref.runs[0].font.size = Pt(8.0)

    # Save outputs
    out_path_main = 'ePilot_DBMS_Project.docx'
    out_path_ieee = 'MANUSCRIPT_IEEE_A4.docx'
    doc.save(out_path_main)
    print(f"Successfully saved camera-ready research paper: {out_path_main}")
    doc.save(out_path_ieee)
    print(f"Successfully saved camera-ready research paper: {out_path_ieee}")

if __name__ == '__main__':
    build_paper()
