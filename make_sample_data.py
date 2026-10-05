"""Tiny demo CSV so the pipeline runs end to end. Replace with real datasets!"""
import pandas as pd
real = ["Government announces new budget allocation for rural education and healthcare",
        "Reserve Bank of India keeps repo rate unchanged after monetary policy meeting",
        "सरकार ने ग्रामीण शिक्षा और स्वास्थ्य के लिए नया बजट घोषित किया",
        "भारतीय रिजर्व बैंक ने मौद्रिक नीति बैठक के बाद रेपो दर में बदलाव नहीं किया",
        "राज्य सरकारने ग्रामीण शिक्षण आणि आरोग्यासाठी नवीन अर्थसंकल्प जाहीर केला",
        "रिझर्व्ह बँकेने पतधोरण बैठकीनंतर रेपो दरात कोणताही बदल केला नाही"]
fake = ["Shocking secret cure doctors don't want you to know forward to everyone immediately",
        "Aliens landed in capital city government hiding truth share before it is deleted",
        "चौंकाने वाला सच डॉक्टर नहीं चाहते आप जानें यह इलाज तुरंत सबको शेयर करें",
        "सरकार छुपा रही है सच एलियन उतरे शहर में डिलीट होने से पहले शेयर करें",
        "धक्कादायक सत्य डॉक्टर तुम्हाला सांगत नाहीत हा उपाय ताबडतोब सर्वांना शेअर करा",
        "सरकार सत्य लपवत आहे एलियन शहरात उतरले डिलीट होण्यापूर्वी शेअर करा"]
rows = [(t, 1) for t in real * 10] + [(t, 0) for t in fake * 10]
pd.DataFrame(rows, columns=["text", "label"]).to_csv("data/news.csv", index=False)
