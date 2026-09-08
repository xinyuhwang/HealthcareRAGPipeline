"""
Synthetic healthcare corpus and graded evaluation set.

IMPORTANT - licensing & privacy note
------------------------------------
All passages below are SYNTHETIC. They were written from scratch to *resemble*
the register of public patient-education / clinical-summary material, but they
are not copied from, or verbatim excerpts of, any copyrighted or licensed
source, and they contain no real patient data (no PHI). They exist only to give
the retrieval pipeline a topic-diverse knowledge base to search over. They are
NOT clinical guidance and must not be used as such.

Design of the corpus (why it looks like this)
---------------------------------------------
A retrieval eval is only as informative as its hard negatives. A corpus of one
obviously-matching passage per query makes every retriever look good and makes
reranking unmeasurable. So this corpus is built with deliberate *distractor
families*: clusters of passages that share the query's surface vocabulary but
answer a different question. For example, for "first-line drug for type 2
diabetes" the corpus also contains first-line therapy for type 1 diabetes,
second-line therapy after metformin, and first-line therapy for gestational
diabetes. A lexical retriever cannot separate these; that separation is what
the dense retriever and the cross-encoder are being measured on.

Graded relevance
----------------
Each query carries a mapping of doc_id -> gain:

    2 = directly answers the question (the passage a human would cite)
    1 = partially relevant / useful supporting context, but not the answer
    0 = irrelevant (omitted from the mapping)

Binary metrics (Recall@k, Precision@k, MRR) treat gain >= RELEVANT_THRESHOLD as
relevant. NDCG@k uses the grades directly, which is the reason for having them:
it distinguishes "found the right passage" from "found something adjacent".

Unanswerable queries
--------------------
Queries with `answerable=False` have NO relevant passage in the corpus. They
are not retrieval failures - they are abstention tests. A correct system
retrieves weak matches and then declines to answer rather than grounding a
clinical claim in an unrelated passage. They are excluded from retrieval
metrics (undefined recall) and scored separately in the answer evaluation.

Ground truth was written against passage *content*, before any retriever was
run, and every query is labeled with the answer the passage actually contains.
"""

RELEVANT_THRESHOLD = 1

# ---------------------------------------------------------------------------
# CORPUS
# ---------------------------------------------------------------------------
CORPUS = [
    # --- Diabetes (DM) -----------------------------------------------------
    {"id": "DM01", "topic": "diabetes", "text": "Type 2 diabetes is diagnosed with a fasting plasma glucose of 126 mg/dL or higher, an HbA1c of 6.5 percent or higher, or a two-hour value of 200 mg/dL or higher on an oral glucose tolerance test, each confirmed on a second occasion."},
    {"id": "DM02", "topic": "diabetes", "text": "Prediabetes describes intermediate glucose values that do not yet meet the threshold for diabetes: a fasting glucose of 100 to 125 mg/dL or an HbA1c of 5.7 to 6.4 percent. It signals elevated risk and warrants lifestyle intervention."},
    {"id": "DM03", "topic": "diabetes", "text": "Type 1 diabetes results from autoimmune destruction of pancreatic beta cells and presents with absolute insulin deficiency, often in younger patients with rapid weight loss. Type 2 diabetes is driven primarily by insulin resistance."},
    {"id": "DM04", "topic": "diabetes", "text": "First-line pharmacologic therapy for most adults with type 2 diabetes is metformin, started alongside lifestyle changes such as dietary modification, weight management, and increased physical activity."},
    {"id": "DM05", "topic": "diabetes", "text": "First-line therapy for type 1 diabetes is insulin replacement, delivered either as multiple daily injections of basal and mealtime insulin or by continuous subcutaneous infusion pump. Oral agents such as metformin do not replace insulin in type 1 disease."},
    {"id": "DM06", "topic": "diabetes", "text": "When metformin alone does not achieve the glucose target, second-line options added on top of it include SGLT2 inhibitors, GLP-1 receptor agonists, DPP-4 inhibitors, or sulfonylureas, selected according to cardiovascular and kidney comorbidity."},
    {"id": "DM07", "topic": "diabetes", "text": "Gestational diabetes is managed first with medical nutrition therapy and glucose self-monitoring; when targets are not met, insulin is the preferred pharmacologic agent during pregnancy."},
    {"id": "DM08", "topic": "diabetes", "text": "Metformin commonly causes gastrointestinal upset that improves with slow dose escalation or an extended-release formulation. It is avoided at severely reduced kidney function because of the risk of lactic acidosis."},
    {"id": "DM09", "topic": "diabetes", "text": "Mild hypoglycemia in a conscious patient is treated with the rule of 15: give 15 grams of fast-acting carbohydrate such as juice or glucose tablets, recheck the glucose after 15 minutes, and repeat if it remains below 70 mg/dL."},
    {"id": "DM10", "topic": "diabetes", "text": "Severe hypoglycemia with confusion, seizure, or loss of consciousness is an emergency treated with injectable or intranasal glucagon, or intravenous dextrose in a monitored setting. Oral carbohydrate must not be forced on an unresponsive patient."},
    {"id": "DM11", "topic": "diabetes", "text": "Adults with diabetes should have a dilated retinal examination at diagnosis and at least every one to two years thereafter, because early diabetic retinopathy is asymptomatic and treatable when detected."},
    {"id": "DM12", "topic": "diabetes", "text": "Diabetic kidney disease is screened for annually with a spot urine albumin-to-creatinine ratio together with an estimated glomerular filtration rate; persistent albuminuria on two of three samples establishes the diagnosis."},
    {"id": "DM13", "topic": "diabetes", "text": "A comprehensive diabetic foot examination includes inspection for ulcers and calluses, pedal pulses, and testing protective sensation with a 10-gram monofilament. Loss of sensation marks a foot at high risk of ulceration."},
    {"id": "DM14", "topic": "diabetes", "text": "Diabetic ketoacidosis presents with hyperglycemia, ketosis, and metabolic acidosis, causing nausea, abdominal pain, deep rapid breathing, and a fruity breath odor. It requires urgent intravenous fluids, insulin, and electrolyte correction."},
    {"id": "DM15", "topic": "diabetes", "text": "A common glycemic target for many non-pregnant adults is an HbA1c below 7 percent, relaxed toward 8 percent in older adults with limited life expectancy, advanced complications, or a history of severe hypoglycemia."},

    # --- Hypertension (HT) -------------------------------------------------
    {"id": "HT01", "topic": "hypertension", "text": "Blood pressure is categorized as normal below 120/80 mmHg, elevated at 120 to 129 systolic with diastolic under 80, and stage 1 hypertension at 130 to 139 systolic or 80 to 89 diastolic."},
    {"id": "HT02", "topic": "hypertension", "text": "Stage 2 hypertension is a systolic pressure of 140 mmHg or higher or a diastolic pressure of 90 mmHg or higher, and generally warrants starting two antihypertensive agents from different classes alongside lifestyle change."},
    {"id": "HT03", "topic": "hypertension", "text": "A hypertensive emergency is a severe pressure elevation, typically above 180/120 mmHg, accompanied by acute damage to the brain, heart, kidneys, or eyes, and requires immediate intravenous treatment. Without organ damage the same reading is a hypertensive urgency, managed with oral agents."},
    {"id": "HT04", "topic": "hypertension", "text": "Initial management of stage 1 hypertension in low-risk patients is lifestyle modification alone for three to six months, with drug therapy added if the pressure remains above target at reassessment."},
    {"id": "HT05", "topic": "hypertension", "text": "The DASH eating pattern emphasizes vegetables, fruit, whole grains, and low-fat dairy while limiting saturated fat, and lowers systolic pressure by roughly 8 to 11 mmHg. Restricting sodium to under 1,500 mg daily adds a further reduction."},
    {"id": "HT06", "topic": "hypertension", "text": "Thiazide-type diuretics such as hydrochlorothiazide and chlorthalidone reduce blood pressure by increasing urinary sodium and water excretion. Patients often call them water pills. Electrolytes should be checked after initiation because of hypokalemia and hyponatremia."},
    {"id": "HT07", "topic": "hypertension", "text": "ACE inhibitors such as lisinopril lower blood pressure by blocking conversion of angiotensin I to angiotensin II. A dry persistent cough affects up to one in ten patients, and angioedema is a rare but serious adverse effect."},
    {"id": "HT08", "topic": "hypertension", "text": "Angiotensin receptor blockers such as losartan provide comparable blood pressure lowering to ACE inhibitors without the bradykinin-mediated cough, and are the usual substitute when an ACE inhibitor cough cannot be tolerated."},
    {"id": "HT09", "topic": "hypertension", "text": "Dihydropyridine calcium channel blockers such as amlodipine relax arterial smooth muscle and are effective first-line agents, particularly in older adults and Black patients. Dose-dependent ankle edema is the most common side effect."},
    {"id": "HT10", "topic": "hypertension", "text": "Beta blockers are not recommended as initial monotherapy for uncomplicated hypertension because they prevent fewer strokes than other classes. They remain indicated when there is a separate reason such as heart failure, prior myocardial infarction, or atrial fibrillation."},
    {"id": "HT11", "topic": "hypertension", "text": "Home blood pressure monitoring uses a validated upper-arm cuff, two readings each morning and evening for seven days, discarding day one and averaging the rest. It predicts cardiovascular outcomes better than isolated office readings."},
    {"id": "HT12", "topic": "hypertension", "text": "White-coat hypertension is an elevated office reading with normal out-of-office averages, and is confirmed by ambulatory or home monitoring before committing a patient to lifelong drug therapy. Masked hypertension is the reverse pattern."},
    {"id": "HT13", "topic": "hypertension", "text": "Resistant hypertension is a pressure above target despite three agents including a diuretic at optimal doses. Evaluation covers adherence, sodium intake, interfering drugs such as NSAIDs, and secondary causes including primary aldosteronism and obstructive sleep apnea."},
    {"id": "HT14", "topic": "hypertension", "text": "Uncontrolled hypertension is a leading modifiable risk factor for stroke, heart failure, myocardial infarction, and chronic kidney disease, and the damage accrues silently over years, which is why consistent follow-up matters even in asymptomatic patients."},

    # --- Asthma / COPD (AS) ------------------------------------------------
    {"id": "AS01", "topic": "asthma", "text": "Asthma control is assessed over the previous four weeks by daytime symptom frequency, any night waking from asthma, reliever use, and activity limitation. Well-controlled asthma has none or at most one of these features."},
    {"id": "AS02", "topic": "asthma", "text": "Inhaled corticosteroids are the preferred long-term controller for persistent asthma. They reduce airway inflammation, symptom burden, and exacerbation frequency, and must be taken daily to be effective rather than only when symptomatic."},
    {"id": "AS03", "topic": "asthma", "text": "A short-acting beta agonist such as albuterol is the rescue medication for sudden asthma symptoms, opening the airways within minutes. Patients often call the device a puffer or rescue inhaler."},
    {"id": "AS04", "topic": "asthma", "text": "A long-acting beta agonist must never be used as sole asthma therapy without an inhaled corticosteroid, because monotherapy increases the risk of severe exacerbations and asthma-related death."},
    {"id": "AS05", "topic": "asthma", "text": "Combination inhaled corticosteroid-formoterol used as needed is now preferred over a short-acting beta agonist alone for mild asthma, and can serve as both the daily controller and the reliever in a single maintenance and reliever regimen."},
    {"id": "AS06", "topic": "asthma", "text": "Peak expiratory flow monitoring tracks airway obstruction against the patient's personal best, and a falling value can signal deterioration before symptoms are noticeable, which is useful for patients who perceive obstruction poorly."},
    {"id": "AS07", "topic": "asthma", "text": "A written asthma action plan divides status into green, yellow, and red zones by symptoms and peak flow, and states exactly which medication to change, at what dose, and when to seek urgent care in each zone."},
    {"id": "AS08", "topic": "asthma", "text": "Using a valved holding chamber, or spacer, with a metered-dose inhaler increases lung deposition and reduces oral thrush. Technique review at each visit matters because incorrect inhaler use is a frequent cause of apparent treatment failure."},
    {"id": "AS09", "topic": "asthma", "text": "Exercise-induced bronchoconstriction is prevented by inhaling a short-acting beta agonist 10 to 15 minutes before activity, together with a warm-up period. Frequent exercise symptoms suggest the underlying asthma is inadequately controlled."},
    {"id": "AS10", "topic": "asthma", "text": "A severe asthma exacerbation is treated with repeated short-acting bronchodilator dosing, systemic corticosteroids for five to seven days, and controlled oxygen. Inability to speak in full sentences or a silent chest indicates life-threatening obstruction."},
    {"id": "AS11", "topic": "copd", "text": "Chronic obstructive pulmonary disease is distinguished from asthma by a smoking or exposure history, onset after age 40, and persistent airflow limitation on spirometry that does not fully reverse with a bronchodilator."},
    {"id": "AS12", "topic": "copd", "text": "Maintenance therapy for COPD is built on long-acting bronchodilators, a LAMA or a LABA or both, with inhaled corticosteroids added for frequent exacerbations or eosinophilia. Pulmonary rehabilitation and smoking cessation alter the disease course most."},

    # --- Infection / antibiotics (ID) --------------------------------------
    {"id": "ID01", "topic": "infection", "text": "Most upper respiratory infections are viral, so antibiotics neither shorten the illness nor prevent complications and expose the patient to side effects and resistance. Treatment is symptomatic: fluids, rest, and analgesia."},
    {"id": "ID02", "topic": "infection", "text": "Streptococcal pharyngitis is suggested by fever, tonsillar exudate, tender anterior cervical nodes, and absence of cough, and should be confirmed by rapid antigen test or culture before treating, because clinical features alone are unreliable."},
    {"id": "ID03", "topic": "infection", "text": "Confirmed streptococcal pharyngitis is treated with penicillin or amoxicillin for ten days, which shortens symptoms modestly and prevents rheumatic fever. Completing the full course matters even though symptoms resolve sooner."},
    {"id": "ID04", "topic": "infection", "text": "Acute bacterial sinusitis is distinguished from a viral course by symptoms persisting beyond ten days without improvement, severe symptoms with purulent discharge and fever, or a double worsening after initial improvement."},
    {"id": "ID05", "topic": "infection", "text": "Uncomplicated cystitis in otherwise healthy non-pregnant women is treated with a short course of nitrofurantoin for five days, or trimethoprim-sulfamethoxazole for three days where local resistance is low."},
    {"id": "ID06", "topic": "infection", "text": "Pyelonephritis is an upper urinary tract infection with flank pain, fever, and systemic illness, and needs a longer antibiotic course than cystitis plus urine culture, with intravenous therapy and admission if the patient cannot keep fluids down."},
    {"id": "ID07", "topic": "infection", "text": "Asymptomatic bacteriuria is bacteria in the urine without urinary symptoms, and should not be treated with antibiotics in most adults because treatment causes harm without benefit. Pregnancy and planned urologic surgery are the main exceptions."},
    {"id": "ID08", "topic": "infection", "text": "Cellulitis presents as spreading skin warmth, redness, swelling, and tenderness with poorly demarcated borders, and is treated with antibiotics covering streptococci and, where purulence or abscess suggests it, Staphylococcus aureus."},
    {"id": "ID09", "topic": "infection", "text": "Sepsis is suspected when infection is accompanied by fever or hypothermia, tachycardia, tachypnea, altered mental status, or hypotension, and it requires immediate evaluation because deterioration can be rapid."},
    {"id": "ID10", "topic": "infection", "text": "In suspected sepsis, blood cultures are drawn and broad-spectrum antibiotics given within the first hour alongside intravenous fluid resuscitation, because each hour of delay to effective antibiotics increases mortality."},
    {"id": "ID11", "topic": "infection", "text": "Finishing the antibiotic course as prescribed, rather than stopping when symptoms improve, reduces relapse and the emergence of resistant organisms. Shorter evidence-based courses are preferred over unnecessarily long ones."},
    {"id": "ID12", "topic": "infection", "text": "Clostridioides difficile infection causes watery diarrhea, cramping, and fever days to weeks after antibiotic exposure disrupts the gut flora. It is diagnosed by stool testing and treated with oral vancomycin or fidaxomicin."},
    {"id": "ID13", "topic": "infection", "text": "A reported penicillin allergy is often not a true allergy, and most patients with a rash-only history tolerate cephalosporins. Formal allergy testing allows first-line therapy to be restored rather than defaulting to broader alternatives."},
    {"id": "ID14", "topic": "infection", "text": "Hand hygiene with soap and water or alcohol-based rub before and after every patient contact is the single most effective measure to prevent healthcare-associated infection. Soap and water is required for visibly soiled hands and for C. difficile spores."},

    # --- Pediatric fever (PF) ----------------------------------------------
    {"id": "PF01", "topic": "pediatric_fever", "text": "In an infant younger than three months, a rectal temperature of 100.4 degrees Fahrenheit or 38 degrees Celsius or higher is a fever requiring prompt evaluation, because serious bacterial infection can occur with few other signs at this age."},
    {"id": "PF02", "topic": "pediatric_fever", "text": "In a well-appearing child between three months and three years, the height of the fever correlates poorly with the seriousness of the illness. Assessment centers on overall appearance, activity, feeding, and hydration rather than the number itself."},
    {"id": "PF03", "topic": "pediatric_fever", "text": "Acetaminophen and ibuprofen reduce fever-related discomfort in children and are dosed by body weight rather than age. Ibuprofen is avoided under six months of age. The aim is comfort, not normalizing the temperature."},
    {"id": "PF04", "topic": "pediatric_fever", "text": "Aspirin is avoided for fever in children and adolescents because of its association with Reye syndrome, a rare but serious encephalopathy with liver dysfunction that can follow viral illness."},
    {"id": "PF05", "topic": "pediatric_fever", "text": "A simple febrile seizure is a brief generalized convulsion during fever in a child between six months and five years, with no lasting neurologic effect. It does not require long-term antiseizure medication, but a first episode warrants evaluation."},
    {"id": "PF06", "topic": "pediatric_fever", "text": "Red flags in a febrile child include lethargy or inconsolability, difficulty breathing or grunting, a non-blanching rash, a stiff neck, persistent vomiting, or signs of dehydration such as no wet diapers for many hours."},
    {"id": "PF07", "topic": "pediatric_fever", "text": "Maintaining hydration during a febrile illness matters more than lowering the temperature. Small frequent volumes of fluid or oral rehydration solution are offered, and reduced urine output signals the need for review."},
    {"id": "PF08", "topic": "pediatric_fever", "text": "A low-grade fever within one to two days of an immunization is a common expected reaction and does not require evaluation on its own. Fever beginning several days later is more likely a separate infection."},

    # --- Vaccination (VX) --------------------------------------------------
    {"id": "VX01", "topic": "vaccination", "text": "The routine childhood immunization schedule covers measles, mumps, and rubella, diphtheria, tetanus, and pertussis, polio, hepatitis B, varicella, Haemophilus influenzae type b, pneumococcal disease, and rotavirus, given in a defined series from birth through adolescence."},
    {"id": "VX02", "topic": "vaccination", "text": "Annual influenza vaccination is recommended for everyone aged six months and older, ideally before influenza begins circulating locally, with rare exceptions based on a specific medical contraindication."},
    {"id": "VX03", "topic": "vaccination", "text": "A catch-up immunization schedule lets a child or adult who has missed doses resume the series from where it stopped, using minimum intervals between doses. The series is not restarted from the beginning regardless of the delay."},
    {"id": "VX04", "topic": "vaccination", "text": "Common vaccine reactions are injection-site soreness, low-grade fever, headache, and fatigue, typically resolving within one to two days. They reflect an expected immune response rather than infection."},
    {"id": "VX05", "topic": "vaccination", "text": "Live attenuated vaccines such as MMR and varicella are contraindicated in significant immunocompromise and in pregnancy. Inactivated vaccines carry no such restriction and are given normally to these groups."},
    {"id": "VX06", "topic": "vaccination", "text": "HPV vaccination is routinely given at age 11 to 12 years, as two doses when started before age 15 and three doses when started later, and prevents cervical and several other HPV-associated cancers."},
    {"id": "VX07", "topic": "vaccination", "text": "Pneumococcal vaccination is recommended for adults aged 65 and older and for younger adults with chronic heart, lung, liver, or kidney disease, diabetes, smoking, or immunocompromise."},
    {"id": "VX08", "topic": "vaccination", "text": "Tdap is given during every pregnancy, preferably between 27 and 36 weeks, so that maternal antibodies cross the placenta and protect the newborn against pertussis before the infant's own vaccination begins."},
    {"id": "VX09", "topic": "vaccination", "text": "A history of egg allergy, including severe allergy, is no longer a reason to withhold influenza vaccine or to require extended observation. Only a documented severe reaction to a previous dose of the same vaccine is a contraindication."},

    # --- Mental health (MH) ------------------------------------------------
    {"id": "MH01", "topic": "mental_health", "text": "The PHQ-9 is a nine-item self-report questionnaire used in primary care to screen for depression and grade its severity, scoring each of the nine DSM criteria from zero to three over the past two weeks."},
    {"id": "MH02", "topic": "mental_health", "text": "The PHQ-2 is a two-item pre-screen asking about depressed mood and loss of interest. A positive PHQ-2 is followed by the full PHQ-9 rather than treated as a diagnosis in itself."},
    {"id": "MH03", "topic": "mental_health", "text": "The GAD-7 is a seven-item instrument that screens for and grades the severity of generalized anxiety symptoms over the past two weeks, with cut-points marking mild, moderate, and severe anxiety."},
    {"id": "MH04", "topic": "mental_health", "text": "A patient endorsing thoughts of self-harm on a screening questionnaire, such as item nine of the PHQ-9, must have a direct suicide risk assessment before leaving the visit, covering intent, plan, means, and protective factors, with a safety plan documented."},
    {"id": "MH05", "topic": "mental_health", "text": "Selective serotonin reuptake inhibitors are the usual first-line medication for moderate to severe depression, with four to six weeks at an adequate dose needed before judging response."},
    {"id": "MH06", "topic": "mental_health", "text": "Cognitive behavioral therapy is as effective as medication for mild to moderate depression, and combining psychotherapy with an antidepressant outperforms either alone in more severe or recurrent illness."},
    {"id": "MH07", "topic": "mental_health", "text": "Screening for a history of mania or hypomania before starting an antidepressant is important, because unopposed antidepressant therapy in bipolar disorder can precipitate a manic switch."},
    {"id": "MH08", "topic": "mental_health", "text": "The AUDIT-C is a three-question screen for unhealthy alcohol use covering frequency, typical quantity, and binge episodes, and a positive result prompts a fuller assessment and brief counseling intervention."},
    {"id": "MH09", "topic": "mental_health", "text": "The Edinburgh Postnatal Depression Scale is the standard screen for perinatal depression, applied during pregnancy and at postpartum visits, and it deliberately omits somatic items that overlap with normal recovery after birth."},

    # --- Pregnancy (PG) ----------------------------------------------------
    {"id": "PG01", "topic": "pregnancy", "text": "Routine prenatal visits in a low-risk pregnancy occur about every four weeks until 28 weeks, every two weeks from 28 to 36 weeks, and weekly from 36 weeks until delivery."},
    {"id": "PG02", "topic": "pregnancy", "text": "Screening for gestational diabetes is performed at 24 to 28 weeks of gestation with a glucose challenge, and earlier in pregnancy when risk factors such as obesity, prior gestational diabetes, or a strong family history are present."},
    {"id": "PG03", "topic": "pregnancy", "text": "Folic acid supplementation of 400 to 800 micrograms daily, begun at least a month before conception and continued through early pregnancy, substantially reduces the risk of neural tube defects."},
    {"id": "PG04", "topic": "pregnancy", "text": "Pregnancy symptoms needing urgent evaluation include vaginal bleeding, leaking fluid, severe or persistent headache, visual disturbance, right upper quadrant pain, and reduced fetal movement after 28 weeks."},
    {"id": "PG05", "topic": "pregnancy", "text": "Preeclampsia is new hypertension at or after 20 weeks of gestation with proteinuria or evidence of organ involvement such as low platelets, abnormal liver enzymes, or visual symptoms, and is definitively managed by delivery."},
    {"id": "PG06", "topic": "pregnancy", "text": "Iron deficiency anemia is common in pregnancy because plasma volume expands faster than red cell mass. Hemoglobin is checked at the first visit and again in the third trimester, with oral iron given for confirmed deficiency."},
    {"id": "PG07", "topic": "pregnancy", "text": "ACE inhibitors and angiotensin receptor blockers are contraindicated in pregnancy because they cause fetal kidney injury and oligohydramnios. Labetalol, nifedipine, and methyldopa are the preferred antihypertensives instead."},
    {"id": "PG08", "topic": "pregnancy", "text": "Postpartum hemorrhage is blood loss of 1,000 mL or more after delivery, most often from uterine atony, and is managed with uterine massage, uterotonic drugs such as oxytocin, and escalation to surgical measures if bleeding continues."},
    {"id": "PG09", "topic": "pregnancy", "text": "Recommended weight gain in pregnancy depends on the pre-pregnancy body mass index, ranging from about 28 to 40 pounds for underweight women down to 11 to 20 pounds for those with obesity."},
    {"id": "PG10", "topic": "pregnancy", "text": "Moderate-intensity physical activity of at least 150 minutes per week is encouraged in uncomplicated pregnancy, avoiding contact sports and activities with a fall risk. Warning signs to stop include bleeding, contractions, and chest pain."},

    # --- Cardiac / lipids (CV) ---------------------------------------------
    {"id": "CV01", "topic": "cardiac", "text": "A lipid panel reports total cholesterol, LDL cholesterol, HDL cholesterol, and triglycerides. LDL is the primary treatment target, and the panel feeds directly into cardiovascular risk estimation and statin decisions."},
    {"id": "CV02", "topic": "cardiac", "text": "Statins are the first-line drug class for lowering LDL cholesterol, reducing production in the liver by inhibiting HMG-CoA reductase, and they lower the rate of heart attack and stroke in proportion to the LDL reduction achieved."},
    {"id": "CV03", "topic": "cardiac", "text": "Muscle aches are the most frequently reported statin complaint, though blinded trials show most such symptoms are not caused by the drug. Management is a washout and rechallenge, a different statin, or intermittent dosing rather than abandoning treatment."},
    {"id": "CV04", "topic": "cardiac", "text": "When a maximally tolerated statin leaves LDL above target, ezetimibe is added next, followed by a PCSK9 inhibitor for very high-risk patients. These are add-on agents rather than statin replacements."},
    {"id": "CV05", "topic": "cardiac", "text": "A pooled cohort risk calculator combines age, sex, race, total and HDL cholesterol, systolic pressure, treatment status, smoking, and diabetes to estimate ten-year risk of atherosclerotic cardiovascular disease and guide preventive therapy."},
    {"id": "CV06", "topic": "cardiac", "text": "Chest pressure lasting more than a few minutes with shortness of breath, sweating, nausea, or radiation to the arm or jaw may be a myocardial infarction and warrants calling emergency services immediately rather than driving to a clinic."},
    {"id": "CV07", "topic": "cardiac", "text": "Stable angina is predictable exertional chest discomfort relieved within minutes by rest or nitroglycerin. Discomfort that occurs at rest, lasts longer, or is escalating in frequency suggests an acute coronary syndrome instead."},
    {"id": "CV08", "topic": "cardiac", "text": "Low-dose aspirin is an antiplatelet agent given indefinitely after a myocardial infarction or ischemic stroke for secondary prevention. Routine use for primary prevention in low-risk adults is no longer advised because bleeding offsets the benefit."},
    {"id": "CV09", "topic": "cardiac", "text": "Atrial fibrillation raises stroke risk from clot formation in the left atrium, and is treated with an oral anticoagulant, usually a direct oral anticoagulant rather than warfarin, when the CHA2DS2-VASc score indicates it. Antiplatelet drugs like aspirin are not adequate substitutes."},
    {"id": "CV10", "topic": "cardiac", "text": "Heart failure presents with exertional breathlessness, orthopnea, waking at night short of breath, ankle swelling, and fatigue, and is confirmed with natriuretic peptide testing and echocardiography to establish the ejection fraction."},
    {"id": "CV11", "topic": "cardiac", "text": "Heart failure with reduced ejection fraction is treated with four foundational drug classes: an ARNI or ACE inhibitor, a beta blocker, a mineralocorticoid receptor antagonist, and an SGLT2 inhibitor, each started at low dose and titrated up."},

    # --- Stroke (ST) -------------------------------------------------------
    {"id": "ST01", "topic": "stroke", "text": "The FAST mnemonic helps the public recognize stroke: Face drooping, Arm weakness, Speech difficulty, and Time to call emergency services immediately. Noting the time symptoms began determines which treatments remain available."},
    {"id": "ST02", "topic": "stroke", "text": "Intravenous thrombolysis for ischemic stroke is given within 4.5 hours of symptom onset after imaging excludes hemorrhage, and the benefit shrinks with every minute of delay, which is why stroke is treated as a time-critical emergency."},
    {"id": "ST03", "topic": "stroke", "text": "Endovascular thrombectomy mechanically removes a large-vessel clot and can be performed up to 24 hours after onset in selected patients with favorable perfusion imaging, either alongside or instead of thrombolysis."},
    {"id": "ST04", "topic": "stroke", "text": "A transient ischemic attack causes stroke symptoms that resolve completely, usually within an hour, but carries a high risk of a completed stroke in the following days. It requires urgent evaluation, not reassurance."},
    {"id": "ST05", "topic": "stroke", "text": "Secondary prevention after an ischemic stroke combines antiplatelet therapy, high-intensity statin treatment, blood pressure control, and management of diabetes and atrial fibrillation, along with smoking cessation and physical activity."},
    {"id": "ST06", "topic": "stroke", "text": "Carotid imaging is performed after an anterior circulation stroke or TIA, and revascularization is considered for symptomatic stenosis of 70 to 99 percent, ideally within two weeks of the event."},

    # --- Lifestyle / nutrition (LS) ----------------------------------------
    {"id": "LS01", "topic": "nutrition", "text": "General dietary guidance emphasizes vegetables, fruit, whole grains, legumes, nuts, and lean protein, while limiting added sugar, refined grains, sodium, and processed meat. Overall dietary pattern matters more than any single nutrient."},
    {"id": "LS02", "topic": "nutrition", "text": "Adults are advised to accumulate at least 150 minutes of moderate-intensity aerobic activity per week, or 75 minutes of vigorous activity, and any increase from a sedentary baseline yields the largest relative health gain."},
    {"id": "LS03", "topic": "nutrition", "text": "Muscle-strengthening activity involving all major muscle groups is recommended on two or more days per week, in addition to aerobic activity, and helps preserve function and bone density with age."},
    {"id": "LS04", "topic": "nutrition", "text": "Combining behavioral counseling with pharmacotherapy such as nicotine replacement, varenicline, or bupropion roughly doubles the chance of quitting smoking compared with unassisted attempts, and repeated attempts remain worthwhile."},
    {"id": "LS05", "topic": "nutrition", "text": "Alcohol guidance limits intake to two standard drinks per day for men and one for women, with less being better. There is no intake level established as beneficial for cardiovascular health."},
    {"id": "LS06", "topic": "nutrition", "text": "Sodium intake should be kept below 2,300 mg daily for most adults, with a target near 1,500 mg for those with hypertension. Most dietary sodium comes from packaged and restaurant food rather than the salt shaker."},
    {"id": "LS07", "topic": "nutrition", "text": "A weight reduction of 5 to 10 percent of body weight produces meaningful improvements in blood pressure, glucose, and lipids, and is a more achievable and durable target than reaching an ideal body weight."},
    {"id": "LS08", "topic": "nutrition", "text": "Seven to nine hours of sleep per night is recommended for adults, and chronic short sleep is associated with hypertension, obesity, and type 2 diabetes. Persistent snoring with daytime sleepiness should prompt sleep apnea evaluation."},

    # --- Kidney (KD) -------------------------------------------------------
    {"id": "KD01", "topic": "kidney", "text": "Chronic kidney disease is staged by estimated glomerular filtration rate from G1 to G5 and separately by albuminuria category A1 to A3. Both axes together determine monitoring frequency and referral, not eGFR alone."},
    {"id": "KD02", "topic": "kidney", "text": "An ACE inhibitor or angiotensin receptor blocker slows progression of proteinuric chronic kidney disease beyond its blood pressure effect. A creatinine rise of up to 30 percent after initiation is expected and is not a reason to stop."},
    {"id": "KD03", "topic": "kidney", "text": "NSAIDs reduce renal perfusion and should be avoided in chronic kidney disease, as should contrast without precautions and unnecessary aminoglycosides. Medication doses generally need adjustment as eGFR falls."},
    {"id": "KD04", "topic": "kidney", "text": "Dialysis is considered when uremic symptoms, refractory fluid overload, hyperkalemia, or acidosis cannot be managed medically, based on symptoms and biochemistry rather than an eGFR threshold alone."},
    {"id": "KD05", "topic": "kidney", "text": "Acute kidney injury is categorized as prerenal from hypoperfusion, intrinsic from tubular or glomerular damage, or postrenal from obstruction, and initial workup covers volume status, medications, urinalysis, and bladder imaging."},
    {"id": "KD06", "topic": "kidney", "text": "Potassium and creatinine are rechecked one to two weeks after starting or increasing an ACE inhibitor, an angiotensin receptor blocker, or a mineralocorticoid receptor antagonist, because hyperkalemia is the main dose-limiting risk."},

    # --- Thyroid (TH) ------------------------------------------------------
    {"id": "TH01", "topic": "thyroid", "text": "Hypothyroidism is screened for with a serum TSH; an elevated TSH with a low free T4 confirms overt primary hypothyroidism. Symptoms include fatigue, cold intolerance, constipation, dry skin, and weight gain."},
    {"id": "TH02", "topic": "thyroid", "text": "Levothyroxine is the treatment for hypothyroidism, taken on an empty stomach separated from calcium and iron, with TSH rechecked six to eight weeks after any dose change because the level takes that long to stabilize."},
    {"id": "TH03", "topic": "thyroid", "text": "Hyperthyroidism presents with weight loss, heat intolerance, palpitations, and tremor, and shows a suppressed TSH with elevated free T4. Graves disease is the usual cause and is treated with antithyroid drugs, radioiodine, or surgery."},
    {"id": "TH04", "topic": "thyroid", "text": "A palpable thyroid nodule is evaluated with TSH and ultrasound, and fine-needle aspiration is directed by the nodule's size and sonographic risk features rather than by palpation alone."},
    {"id": "TH05", "topic": "thyroid", "text": "Subclinical hypothyroidism is a raised TSH with a normal free T4. Treatment is generally reserved for a TSH above 10, or for symptoms, positive antibodies, or pregnancy, and the test is repeated before committing to therapy."},
]

# ---------------------------------------------------------------------------
# EVALUATION QUERIES
# ---------------------------------------------------------------------------
# kind:
#   direct        - question phrased close to the passage's own wording
#   paraphrase    - lay or synonym vocabulary that does not overlap the passage
#                   ("water pill" -> thiazide diuretic); separates dense from lexical
#   distractor    - a near-miss passage shares more surface vocabulary than the
#                   correct one; separates a reranker from raw first-stage recall
#   multi_doc     - several passages contribute, with graded gains
#   unanswerable  - no passage answers this; abstention test
QUERIES = [
    # --- direct ------------------------------------------------------------
    {"query": "How is type 2 diabetes diagnosed?", "kind": "direct",
     "relevant": {"DM01": 2, "DM02": 1}},
    {"query": "What is the first-line medication for type 2 diabetes?", "kind": "direct",
     "relevant": {"DM04": 2, "DM06": 1}},
    {"query": "How often should an adult with diabetes have a retinal exam?", "kind": "direct",
     "relevant": {"DM11": 2}},
    {"query": "What blood pressure range counts as stage 1 hypertension?", "kind": "direct",
     "relevant": {"HT01": 2, "HT02": 1}},
    {"query": "Which drug classes are used first-line for high blood pressure?", "kind": "direct",
     "relevant": {"HT06": 2, "HT07": 2, "HT08": 2, "HT09": 2, "HT10": 1}},
    {"query": "What is the preferred long-term controller medication for persistent asthma?", "kind": "direct",
     "relevant": {"AS02": 2, "AS05": 1}},
    {"query": "How does an asthma action plan work?", "kind": "direct",
     "relevant": {"AS07": 2, "AS06": 1}},
    {"query": "Should antibiotics be prescribed for an upper respiratory infection?", "kind": "direct",
     "relevant": {"ID01": 2}},
    {"query": "How is uncomplicated cystitis treated in healthy women?", "kind": "direct",
     "relevant": {"ID05": 2}},
    {"query": "What are the signs of sepsis?", "kind": "direct",
     "relevant": {"ID09": 2, "ID10": 1}},
    {"query": "What rectal temperature counts as a fever in an infant under three months?", "kind": "direct",
     "relevant": {"PF01": 2}},
    {"query": "Which medicines are used to reduce fever in children?", "kind": "direct",
     "relevant": {"PF03": 2, "PF04": 1}},
    {"query": "Who should receive an annual influenza vaccine?", "kind": "direct",
     "relevant": {"VX02": 2, "VX09": 1}},
    {"query": "What screening questionnaire is used for depression in primary care?", "kind": "direct",
     "relevant": {"MH01": 2, "MH02": 1}},
    {"query": "Which tool screens for generalized anxiety symptoms?", "kind": "direct",
     "relevant": {"MH03": 2}},
    {"query": "How frequently are prenatal visits scheduled in a low-risk pregnancy?", "kind": "direct",
     "relevant": {"PG01": 2}},
    {"query": "When is screening for gestational diabetes performed?", "kind": "direct",
     "relevant": {"PG02": 2, "DM07": 1}},
    {"query": "What does a lipid panel measure?", "kind": "direct",
     "relevant": {"CV01": 2}},
    {"query": "Which drug class is first-line for lowering LDL cholesterol?", "kind": "direct",
     "relevant": {"CV02": 2, "CV04": 1}},
    {"query": "How can the public recognize the symptoms of a stroke?", "kind": "direct",
     "relevant": {"ST01": 2}},
    {"query": "How is chronic kidney disease staged?", "kind": "direct",
     "relevant": {"KD01": 2}},
    {"query": "Which blood test screens for an underactive thyroid?", "kind": "direct",
     "relevant": {"TH01": 2, "TH05": 1}},
    {"query": "How much aerobic physical activity is recommended for adults each week?", "kind": "direct",
     "relevant": {"LS02": 2, "LS03": 1}},
    {"query": "Does counseling combined with medication improve smoking cessation rates?", "kind": "direct",
     "relevant": {"LS04": 2}},
    {"query": "What is the recommended time window for intravenous thrombolysis in ischemic stroke?", "kind": "direct",
     "relevant": {"ST02": 2, "ST03": 1}},
    {"query": "How is severe hypoglycemia with loss of consciousness treated?", "kind": "direct",
     "relevant": {"DM10": 2, "DM09": 1}},
    {"query": "What is resistant hypertension and how is it evaluated?", "kind": "direct",
     "relevant": {"HT13": 2}},
    {"query": "How is a severe asthma exacerbation managed?", "kind": "direct",
     "relevant": {"AS10": 2}},
    {"query": "What are the four foundational drug classes for heart failure with reduced ejection fraction?", "kind": "direct",
     "relevant": {"CV11": 2, "CV10": 1}},
    {"query": "When should dialysis be started in chronic kidney disease?", "kind": "direct",
     "relevant": {"KD04": 2}},

    # --- paraphrase (lay vocabulary, little lexical overlap) ---------------
    {"query": "My sugar levels run high in the morning even before I eat anything. What test confirms it?", "kind": "paraphrase",
     "relevant": {"DM01": 2, "DM02": 1}},
    {"query": "What is the rule for treating a low sugar episode with juice?", "kind": "paraphrase",
     "relevant": {"DM09": 2, "DM10": 1}},
    {"query": "Which water pill is used to bring blood pressure down?", "kind": "paraphrase",
     "relevant": {"HT06": 2}},
    {"query": "My blood pressure tablet gives me a constant dry tickly cough. What can I switch to?", "kind": "paraphrase",
     "relevant": {"HT08": 2, "HT07": 2}},
    {"query": "Why do my ankles swell up since starting the pill for my blood pressure?", "kind": "paraphrase",
     "relevant": {"HT09": 2}},
    {"query": "Which puffer do I grab when I suddenly cannot catch my breath?", "kind": "paraphrase",
     "relevant": {"AS03": 2, "AS07": 1}},
    {"query": "Is there a plastic tube thing that helps the medicine get into the lungs properly?", "kind": "paraphrase",
     "relevant": {"AS08": 2}},
    {"query": "I get wheezy whenever I go running. What do I take beforehand?", "kind": "paraphrase",
     "relevant": {"AS09": 2, "AS03": 1}},
    {"query": "Should I keep taking the pills after the infection feels better?", "kind": "paraphrase",
     "relevant": {"ID11": 2, "ID03": 1}},
    {"query": "I got terrible watery diarrhea a couple of weeks after taking antibiotics.", "kind": "paraphrase",
     "relevant": {"ID12": 2}},
    {"query": "Is the baby aspirin worth taking to prevent a first heart attack if I am healthy?", "kind": "paraphrase",
     "relevant": {"CV08": 2, "CV05": 1}},
    {"query": "I need a blood thinner because my heart beats irregularly. Which kind?", "kind": "paraphrase",
     "relevant": {"CV09": 2}},
    {"query": "My legs ache since I started the cholesterol tablet. Do I have to stop it?", "kind": "paraphrase",
     "relevant": {"CV03": 2}},
    {"query": "I am always cold, tired, and constipated and my skin is dry. What should be checked?", "kind": "paraphrase",
     "relevant": {"TH01": 2}},
    {"query": "How long after changing my thyroid tablet dose should the blood test be repeated?", "kind": "paraphrase",
     "relevant": {"TH02": 2}},
    {"query": "Weakness on one side of the face and slurred words came on suddenly. What do I do?", "kind": "paraphrase",
     "relevant": {"ST01": 2, "ST02": 1}},
    {"query": "The symptoms went away completely after half an hour, so is it still worth being seen?", "kind": "paraphrase",
     "relevant": {"ST04": 2}},
    {"query": "How much salt is too much if my pressure is up?", "kind": "paraphrase",
     "relevant": {"LS06": 2, "HT05": 2}},
    {"query": "Do I need to lose all the extra weight to get any benefit?", "kind": "paraphrase",
     "relevant": {"LS07": 2}},
    {"query": "My toddler is burning up but is playing happily. How worried should I be?", "kind": "paraphrase",
     "relevant": {"PF02": 2, "PF06": 1}},
    {"query": "The baby got a temperature the day after her shots. Is that expected?", "kind": "paraphrase",
     "relevant": {"PF08": 2, "VX04": 2}},
    {"query": "My child missed several vaccine doses. Do we have to start the whole series again?", "kind": "paraphrase",
     "relevant": {"VX03": 2}},
    {"query": "What vitamin should I take before trying to conceive to protect the baby's spine?", "kind": "paraphrase",
     "relevant": {"PG03": 2}},
    {"query": "I am short of breath lying flat and my ankles are swollen in the evening.", "kind": "paraphrase",
     "relevant": {"CV10": 2, "CV11": 1}},
    {"query": "Are painkillers like ibuprofen safe if my kidney function is reduced?", "kind": "paraphrase",
     "relevant": {"KD03": 2}},

    # --- distractor-sensitive (a near-miss shares more vocabulary) ---------
    {"query": "What is first-line treatment for type 1 diabetes?", "kind": "distractor",
     "relevant": {"DM05": 2, "DM03": 1}},
    {"query": "What should be added when metformin alone is not enough?", "kind": "distractor",
     "relevant": {"DM06": 2, "DM04": 1}},
    {"query": "Which glucose-lowering drug is preferred during pregnancy?", "kind": "distractor",
     "relevant": {"DM07": 2, "PG02": 1}},
    {"query": "Which blood pressure medications must be stopped in pregnancy?", "kind": "distractor",
     "relevant": {"PG07": 2}},
    {"query": "Is a reading over 180/120 with chest pain an emergency or an urgency?", "kind": "distractor",
     "relevant": {"HT03": 2, "HT02": 1}},
    {"query": "Can a long-acting beta agonist be used on its own for asthma?", "kind": "distractor",
     "relevant": {"AS04": 2, "AS02": 1}},
    {"query": "How do I tell whether persistent breathlessness is COPD rather than asthma?", "kind": "distractor",
     "relevant": {"AS11": 2, "AS12": 1}},
    {"query": "Should bacteria found in the urine of a patient with no symptoms be treated?", "kind": "distractor",
     "relevant": {"ID07": 2, "ID05": 1}},
    {"query": "How does kidney infection with flank pain and fever differ from a simple bladder infection?", "kind": "distractor",
     "relevant": {"ID06": 2, "ID05": 1}},
    {"query": "Does a sore throat need a swab before antibiotics are given?", "kind": "distractor",
     "relevant": {"ID02": 2, "ID03": 1}},
    {"query": "Why is aspirin avoided for fever in children?", "kind": "distractor",
     "relevant": {"PF04": 2, "PF03": 1}},
    {"query": "Which vaccines cannot be given to someone who is immunocompromised?", "kind": "distractor",
     "relevant": {"VX05": 2, "VX01": 1}},
    {"query": "What comes after a positive two-question depression pre-screen?", "kind": "distractor",
     "relevant": {"MH02": 2, "MH01": 2}},
    {"query": "What must be ruled out before starting an antidepressant?", "kind": "distractor",
     "relevant": {"MH07": 2, "MH05": 1}},
    {"query": "Which depression screening tool is designed for use around childbirth?", "kind": "distractor",
     "relevant": {"MH09": 2, "MH01": 1}},
    {"query": "Is exertional chest discomfort relieved by rest an acute coronary syndrome?", "kind": "distractor",
     "relevant": {"CV07": 2, "CV06": 1}},
    {"query": "What is added when a maximum-dose statin still leaves LDL too high?", "kind": "distractor",
     "relevant": {"CV04": 2, "CV02": 1}},
    {"query": "Is a rise in creatinine after starting an ACE inhibitor a reason to stop it?", "kind": "distractor",
     "relevant": {"KD02": 2, "KD06": 1}},
    {"query": "When is treatment indicated for a raised TSH with a normal free T4?", "kind": "distractor",
     "relevant": {"TH05": 2, "TH01": 1}},
    {"query": "Which beta blocker situations still justify their use in hypertension?", "kind": "distractor",
     "relevant": {"HT10": 2}},

    # --- multi-document ----------------------------------------------------
    {"query": "What lifestyle changes lower blood pressure?", "kind": "multi_doc",
     "relevant": {"HT04": 2, "HT05": 2, "LS06": 2, "LS02": 1, "LS07": 1, "LS05": 1, "LS01": 1}},
    {"query": "Why does uncontrolled high blood pressure matter if I feel fine?", "kind": "multi_doc",
     "relevant": {"HT14": 2, "KD01": 1, "ST05": 1, "CV10": 1}},
    {"query": "What is the full set of routine annual screenings for a patient with diabetes?", "kind": "multi_doc",
     "relevant": {"DM11": 2, "DM12": 2, "DM13": 2, "DM15": 1}},
    {"query": "How should a patient who screens positive for suicidal thoughts be handled?", "kind": "multi_doc",
     "relevant": {"MH04": 2, "MH01": 1}},
    {"query": "What should be done in the first hour for a patient with suspected sepsis?", "kind": "multi_doc",
     "relevant": {"ID10": 2, "ID09": 2}},
    {"query": "Which vaccines are recommended during pregnancy?", "kind": "multi_doc",
     "relevant": {"VX08": 2, "VX05": 1, "VX02": 1}},
    {"query": "What are the danger signs in pregnancy and what condition do they suggest?", "kind": "multi_doc",
     "relevant": {"PG04": 2, "PG05": 2}},
    {"query": "How is stroke risk reduced after someone has already had one?", "kind": "multi_doc",
     "relevant": {"ST05": 2, "ST06": 1, "CV08": 1, "CV09": 1}},
    {"query": "What monitoring is needed after starting an ACE inhibitor?", "kind": "multi_doc",
     "relevant": {"KD06": 2, "KD02": 1, "HT07": 1}},
    {"query": "How is asthma control assessed and tracked over time?", "kind": "multi_doc",
     "relevant": {"AS01": 2, "AS06": 2, "AS07": 1}},

    # --- unanswerable (abstention tests; no relevant passage exists) -------
    {"query": "What is the recommended chemotherapy regimen for stage III colon cancer?", "kind": "unanswerable",
     "relevant": {}, "answerable": False},
    {"query": "How is rheumatoid arthritis distinguished from psoriatic arthritis on imaging?", "kind": "unanswerable",
     "relevant": {}, "answerable": False},
    {"query": "What is the surgical approach for a torn anterior cruciate ligament?", "kind": "unanswerable",
     "relevant": {}, "answerable": False},
    {"query": "Which antiretroviral regimen is first-line for newly diagnosed HIV?", "kind": "unanswerable",
     "relevant": {}, "answerable": False},
    {"query": "How is glaucoma intraocular pressure managed with eye drops?", "kind": "unanswerable",
     "relevant": {}, "answerable": False},
    {"query": "What is the ventilator strategy for acute respiratory distress syndrome?", "kind": "unanswerable",
     "relevant": {}, "answerable": False},
    {"query": "How should a patient with cirrhosis and ascites be treated?", "kind": "unanswerable",
     "relevant": {}, "answerable": False},
    {"query": "What is the dosing schedule for insulin in a patient with a pancreatic tumor resection?", "kind": "unanswerable",
     "relevant": {}, "answerable": False},
    {"query": "Which biologic is preferred for moderate to severe ulcerative colitis?", "kind": "unanswerable",
     "relevant": {}, "answerable": False},
    {"query": "How is multiple sclerosis relapse treated acutely?", "kind": "unanswerable",
     "relevant": {}, "answerable": False},
]

# Default `answerable=True` for every query that does not say otherwise.
for _q in QUERIES:
    _q.setdefault("answerable", True)
del _q


def answerable_queries():
    """Queries used for retrieval metrics (recall is undefined without a target)."""
    return [q for q in QUERIES if q["answerable"]]


def unanswerable_queries():
    """Queries used only for abstention / failure-handling evaluation."""
    return [q for q in QUERIES if not q["answerable"]]


def corpus_ids():
    return [d["id"] for d in CORPUS]


def validate():
    """Self-check the eval set. Raises AssertionError on any inconsistency."""
    ids = corpus_ids()
    assert len(ids) == len(set(ids)), "duplicate document ids in CORPUS"
    id_set = set(ids)

    for q in QUERIES:
        for doc_id, gain in q["relevant"].items():
            assert doc_id in id_set, f"query {q['query']!r} labels unknown id {doc_id}"
            assert gain in (1, 2), f"query {q['query']!r} has invalid gain {gain}"
        if q["answerable"]:
            assert any(g == 2 for g in q["relevant"].values()), (
                f"answerable query {q['query']!r} has no gain-2 passage"
            )
        else:
            assert not q["relevant"], (
                f"unanswerable query {q['query']!r} must have no labels"
            )
    return {
        "documents": len(CORPUS),
        "topics": len({d["topic"] for d in CORPUS}),
        "queries": len(QUERIES),
        "answerable": len(answerable_queries()),
        "unanswerable": len(unanswerable_queries()),
        "labels": sum(len(q["relevant"]) for q in QUERIES),
    }


if __name__ == "__main__":
    from collections import Counter

    stats = validate()
    for key, value in stats.items():
        print(f"{key:>14}: {value}")
    print(f"{'by kind':>14}: {dict(Counter(q['kind'] for q in QUERIES))}")
