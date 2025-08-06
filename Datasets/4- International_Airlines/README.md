# 📊 Datasets – International Air Traffic (Australia)

## 🧾 Description

This dataset contains detailed records of **passenger, freight, and mail traffic** for international scheduled airline operations **to and from Australia**. Each row represents a city-pair direction (inbound/outbound), aggregated by port and flight number.

### 📁 Columns

| Column                  | Description |
|-------------------------|-------------|
| `AustralianPort`        | Port in Australia (uplift or discharge) |
| `ForeignPort`           | Corresponding foreign port |
| `Port_Country`          | Country of the foreign port |
| `Passengers_In`         | Passengers arriving to Australia |
| `Freight_In_(tonnes)`   | Inbound freight (tonnes) |
| `Mail_In_(tonnes)`      | Inbound mail (tonnes) |
| `Passengers_Out`        | Passengers departing from Australia |
| `Freight_Out_(tonnes)`  | Outbound freight (tonnes) |
| `Mail_Out_(tonnes)`     | Outbound mail (tonnes) |


---

## 🗂 File Info

- Format: CSV
- Rows: +89k 
- Time Span: Includes records over multiple years
- Coverage: Scheduled international air carriers only (no charter data)
- [📥 Download the dataset here](https://www.kaggle.com/datasets/imtkaggleteam/international-airlines-traffic-by-city-pairs/data)


---

## 🧾 Important Notes

- **Code-share caveats**: Some airline data may omit code-shared flights.
- **Home-country bias**: Data may overstate traffic through an airline's home port due to flight number definitions.
- **Missing periods**: Some airlines have gaps in reporting (see full metadata).
- **City-pair reporting shifted in 2003+**: Now based on single flight number segments (not full route).

---

## 📜 License

Creative Commons Attribution 3.0 Australia  
Provided by the **Bureau of Infrastructure and Transport Research Economics (BITRE)**

---

## 🔗 Source

These statistics were collected from scheduled international carriers and processed by BITRE (Australia).

