import pandas as pd

# --------------------------------
# 10 Students Dataset
# --------------------------------
data = {
    "Marks": [85, 60, 45, 90, 70, 40, 88, 55, 65, 30],
    "Attendance": [90, 75, 60, 95, 80, 55, 92, 70, 78, 50],
    "FamilyIncome": [120000, 140000, 200000, 100000, 130000, 300000, 110000, 250000, 145000, 400000]
}

df = pd.DataFrame(data)

print("Dataset:\n", df)

print("\n---- Checking Eligibility for Dataset ----")

for i in range(len(df)):
    if df["Marks"][i] > 45 and df["Attendance"][i] > 75 and df["FamilyIncome"][i] < 150000:
        print("Student", i+1, ": Eligible")
    else:
        print("Student", i+1, ": Not Eligible")

# --------------------------------
# New Student Input
# --------------------------------
print("\n---- New Student Check ----")

marks = int(input("Enter Marks: "))
attendance = int(input("Enter Attendance %: "))
income = int(input("Enter Family Income: "))

if marks > 45 and attendance > 75 and income < 150000:
    print("✅ Student is ELIGIBLE for Scholarship")
else:
    print("❌ Student is NOT ELIGIBLE for Scholarship")
https://elevate-student-way.lovable.app