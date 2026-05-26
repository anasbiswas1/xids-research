# Example LLM Alerts (Llama-3-8B)

## rf_binary_cw — low_trust (SCTS-v2 = 47.6)

**Predicted class:** Attack  **True class:** Normal  **Action:** ENRICH

> Here is the alert:

A potential attack has been detected with a calibrated confidence of 0.63 and a SCTS-v2 trust score of 47.6/100. The top features driving this prediction are src_bytes, dst_bytes, logged_in, flag_SF, and count, all of which decrease the attack-class likelihood. Based on this detection, I recommend enriching the alert with threat intelligence before taking action, as the trust score indicates a low-confidence prediction.

## rf_binary_cw — high_trust (SCTS-v2 = 94.5)

**Predicted class:** Attack  **True class:** Attack  **Action:** BLOCK

> Here is the alert:

A high-confidence attack has been detected with a trust score of 94.5/100. The top features driving this prediction are the increased src_bytes and dst_host_srv_serror_rate, indicating a potential attack. Based on this high-trust prediction, I recommend blocking the traffic immediately to prevent potential harm.

## xgb_binary_cw — low_trust (SCTS-v2 = 35.6)

**Predicted class:** Normal  **True class:** Normal  **Action:** ENRICH

> Here is the alert:

A potential normal network traffic pattern has been detected with a trust score of 35.6/100. The top features driving this prediction are src_bytes, which increases the attack-class likelihood, and dst_bytes, srv_count, protocol_type_tcp, and dst_host_rerror_rate, which all decrease the attack-class likelihood. As the trust score is below the threshold, I recommend ENRICHing the alert to gather more context before clearing. The low trust score warrants further investigation to ensure the traffic is legitimate.

## xgb_binary_cw — high_trust (SCTS-v2 = 87.4)

**Predicted class:** Attack  **True class:** Attack  **Action:** BLOCK

> Here is the alert:

A high-confidence attack has been detected with a trust score of 87.4/100. The top features driving this prediction are src_bytes, count, service_private, flag_S0, and dst_host_srv_serror_rate, all of which increase the likelihood of an attack. Based on this high-trust prediction, I recommend blocking the traffic immediately to prevent potential harm.

## dnn_binary_cw — low_trust (SCTS-v2 = 63.8)

**Predicted class:** Normal  **True class:** Normal  **Action:** MONITOR

> Here is the alert:

A moderate-trust normal prediction has been detected, with a calibrated confidence of 0.67 and a SCTS-v2 trust score of 63.8/100. The top features driving this prediction are the presence of a hot connection, a high dst_host_srv_count, and a dst_host_same_src_port_rate, all of which increase the likelihood of an attack. As a result, I recommend monitoring this activity closely and logging it for further observation, as the model's prediction is not entirely conclusive.

## dnn_binary_cw — high_trust (SCTS-v2 = 92.8)

**Predicted class:** Attack  **True class:** Attack  **Action:** BLOCK

> Here is the alert:

A high-confidence attack has been detected with a trust score of 92.8/100. The top features driving this prediction are the high error rates on the destination host and the same server, indicating a potential attack. Based on this information, I recommend blocking the traffic immediately to prevent further potential damage.

## rf_5class_smote — low_trust (SCTS-v2 = 32.5)

**Predicted class:** Normal  **True class:** DoS  **Action:** ENRICH

> Here is the alert:

A potential security threat has been detected with a calibrated confidence of 0.33 and a trust score of 32.5/100. The top features driving this prediction are dst_host_same_src_port_rate, dst_host_serror_rate, dst_host_srv_count, dst_host_diff_srv_rate, and flag_S0, all of which increase the likelihood of an attack. As the trust score is below the threshold, I recommend ENRICHing the alert to gather more context before clearing it.

## rf_5class_smote — high_trust (SCTS-v2 = 86.6)

**Predicted class:** Normal  **True class:** R2L  **Action:** BENIGN

> Here is the alert:

A normal traffic pattern has been detected with a trust score of 86.6/100, indicating a high level of confidence in the prediction. The top features driving this prediction are src_bytes and dst_host_same_src_port_rate, which both contribute to an increased likelihood of attack-class traffic. Based on this analysis, no further action is required, as the detection model has predicted this traffic as benign.

## xgb_5class_smote — low_trust (SCTS-v2 = 43.3)

**Predicted class:** DoS  **True class:** Probe  **Action:** ENRICH

> Here is the alert:

A potential Denial of Service (DoS) attack has been detected with a calibrated confidence of 0.44. The top features driving this prediction are flag_S0, which increases the attack-class likelihood, and count, dst_host_srv_diff_host_rate, dst_host_serror_rate, and serror_rate, which all decrease the attack-class likelihood. Based on this information, I recommend enriching the alert with threat intelligence before taking action, as the SCTS-v2 trust score is 43.3/100, indicating a low level of confidence.

## xgb_5class_smote — high_trust (SCTS-v2 = 87.4)

**Predicted class:** DoS  **True class:** DoS  **Action:** BLOCK

> Here is the alert:

A high-confidence DoS attack has been detected with a trust score of 87.4/100. The top features driving this prediction are flag_S0 and count, which both increase the likelihood of an attack, with SHAP values of +2.652 and +2.374, respectively. Based on this high-trust prediction, I recommend blocking the traffic immediately to prevent further damage.

## dnn_5class_smote — low_trust (SCTS-v2 = 40.4)

**Predicted class:** Normal  **True class:** Normal  **Action:** ENRICH

> Here is the alert:

A potential security threat has been detected with a calibrated confidence of 0.35 and a trust score of 40.4/100. The top features driving this prediction are service_finger (SHAP +1.208) and hot (SHAP +0.588), which indicate an increased likelihood of an attack. As the trust score is below the threshold, I recommend ENRICHing the alert to gather more context before clearing it.

## dnn_5class_smote — high_trust (SCTS-v2 = 92.4)

**Predicted class:** DoS  **True class:** DoS  **Action:** BLOCK

> Here is the alert:

A high-confidence DoS attack has been detected with a trust score of 92.4/100. The top features driving this prediction are dst_host_rerror_rate and same_srv_rate, which indicate a significant increase in attack-class likelihood. Based on this high-trust prediction, I recommend blocking the traffic immediately to prevent potential damage.

