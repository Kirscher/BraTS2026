# Submission

Synapse wiki: syn74274097/wiki/639606
Modified: 2026-06-02T20:01:39.451Z

This section is for uploading and submitting your prediction files or Docker images. For the short paper submissions, please follow the guidelines in the **Instructions** tab.

### **<font color="#5c94b9">Validation Phase</font>**

Generate a 2-column CSV file with your predictions. For consistency with the Test Phase, please name the CSV file `predictions.csv` and ensure it adheres to the following format:

**Column Name** | **Column Type** | **Accepted Values** | **Example Row in CSV**
----------------|-----------------|---------------------|-----------------------
<font color="#e8762b">`SubjectID`</font> | str | Filename of the digitized tissue section (excluding the .jpg extension) | <li><font color="#e8762b">`val_3fc7f8ced2878bec`</font></li>
<font color="#e8762b">`Prediction `</font> | int | One of 0~9 according to the <font color="#e8762b">`class_map.json`</font>: <br/> `0 = CT` &nbsp;&nbsp; `1 = DM` &nbsp;&nbsp; `2 = IC` <br/> `3 = LI` &nbsp;&nbsp; `4 = MP` &nbsp;&nbsp; `5 = NC` <br/> `6 = PL` &nbsp;&nbsp; `7 = PN` &nbsp;&nbsp; `8 = WM` <br/> `9 = NOTA` | 1

--------------------------------------------------------------------------------------------------------------

### **<font color="#5c94b9">Test Phase</font>**
**<font color="#e8762b">To be updated</font>**






> ⚠️ **Trouble submitting?**
> You must be fully registered and part of a team to make a submission. If you are getting an error message or the widgets are disabled, your registration process is likely incomplete.
