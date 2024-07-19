#!/usr/bin/env python
# coding: utf-8

# # RFM Customer Segmentation 

#     Segmentation : เพื่อแบ่งกลุ่มกลุ่มลูกค้าตามพฤติกรรมการซื้อของลูกค้าในแต่ละ transaction โดยที่  Model นี้อาจเหมาะสมกับสินค้าที่มีการซื้อซ้ำ
#     เช่น อาหาร หรือเครื่องสำอางค์ เสื้อผ้า โดยมีตัวแปรสำคัญที่ใช้ในพิจารณา 3 ตัว คือ 
#             1. Recency(R) คือ จำนวนวันที่ลูกค้าซื้อสินค้าล่าสุด นับจากวันที่เราพิจารณา
#             2. Frequency(F) คือ ความถี่ที่ลูกค้ามาซื้อสินค้า มาซื้อบ่อยแค่ไหน
#             3. Monetary(M) คือ จำนวนเงินลูกค้าซื้อตั้งแต่ transaction แรกจนถึงวันที่เราพิจารณา  ** โปรเจคนี้จะพิจารณาจากกำไร**
#     

# # 1. นำเข้า Library 

# In[161]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")


# # 2. นำเข้า dataset 

# In[162]:


#อ่านไฟล์ csv 
df = pd.read_csv("sales_data.csv")
df


# # 3. เลือกเฉพาะ columns ที่สำคัญ และเปลี่ยนชื่อ columns ให้อ่านเข้าใจง่าย

#     ข้อมูลที่เลือกมาสร้าง RFM Model มีดังนี้ 
# 1. _CustomerID รหัสลูกค้า
# 2. OrderNumber รหัสการสั่งซื้อ
# 3. OrderDate วันที่สั่งซื้อ
# 4. Order Quantity จำนวนชิ้น
# 5. Discount Applied อัตราส่วนลด
# 6. Unit Price ราคาขายต่อชิ้น
# 7. Unit Cost ต้นทุนต่อชิ้น

# In[163]:


# เรียกดูชื่อ clolumns 
df.columns


# In[164]:


#เลือกcolumns ที่จะใช้งาน สำหรับ RFM 
# '_CustomerID', 'OrderNumber', 'OrderDate', 'Order Quantity', 'Discount Applied', 'Unit Price', 'Unit Cost'

df01 = df[['_CustomerID', 'OrderNumber', 'OrderDate', 'Order Quantity', 'Discount Applied', 'Unit Price', 'Unit Cost']]
df01


# In[165]:


#หาค่า Profit
#เก็บค่า Profit ลงไปใน dataframe df01  -->> df01["Profit"]
# Profit = รายรับ - ต้นทุน - ส่วนลด 
df01["Profit"] = (df01["Unit Price"] - df01["Unit Cost"] - df01["Unit Price"]*df01["Discount Applied"])*df01["Order Quantity"]
df01["Profit"]


# In[166]:


df01


# In[167]:


# เลือก columns ที่จำเป็น 
# "_CustomerID", "OrderNumber", "OrderDate", "Profit"
df01 = df01[["_CustomerID", "OrderNumber", "OrderDate", "Profit"]]
df01


# In[168]:


# ตั้งชื่อใช้ เป็น df 
# เปลี่ยนชื่อแต่ละ column ให้อ่านง่าย

df = df01.rename(columns= {
           "_CustomerID" : "Customer_ID",
           "OrderNumber" : "Order_Number",
           "OrderDate" : "Order_Date",
           "Profit" : "Profit"
           })

df


# # 4. Data preparation

# In[169]:


# Check missing values

df.isnull().sum() #ไม่มีค่า null


# In[170]:


#Check data type
df.dtypes


# In[171]:


#เปลี่ยน Order_Date เป็น datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Order_Date"]


# In[172]:


# check ค่า revenue ติดลบมั้ย ขาดทุน
df[df["Profit"] < 0]


# In[173]:


df["Order_Date"].max()


# # 5.คำนวณค่า RFM Metrics และสร้างกรอบวันที่ต้องการวิเคราะห์

# In[174]:


# เนื่องจาก df["Order_Date"].max()  คือ 2020-12-30
# ดังนั้นจึงกำหนด วันที่พิจารณา analyze_date คือ 2021-01-01
analyze_date = pd.to_datetime('2021-01-01')


# In[175]:


df.columns


# In[176]:


# Grouping ลูกค้าที่มีชื่อเดียวกัน (customer_name)
rfm_dataset = df.groupby(["Customer_ID"]).agg({
        "Order_Date" : lambda x :  (analyze_date - x.max()).days ,
        "Order_Number" : lambda x : x.nunique(), #นับ order ที่ไม่ซ้ำกัน
        "Profit" : lambda x : x.sum() 
    })


# In[177]:


rfm_dataset


# In[178]:


# ตรวจ data type ของ columns Order_Date ต้องเป็น int
rfm_dataset.dtypes


# In[179]:


# เปลี่ยนชื่อ columns 
rfm_dataset.rename(
    columns= {
        "Order_Date" : "Recency",
        "Order_Number" : "Frequency",
        "Profit" : "Monetary"
    }, inplace = True)


# In[180]:


rfm_dataset.head(3)


# # 6. สร้าง R F M columns คำนวณ RFM_Score และ RF_Score ตามลำดับ

# In[181]:


#rfm_dataset = rfm_dataset.assign(R = R.values, F = F.values, M = M.values)


# In[182]:


# สร้าง R F M columns 
rfm_dataset["R"] = pd.qcut(rfm_dataset["Recency"], q = 5, labels= range(5, 0, -1)) # [5,4,3,2,1]
rfm_dataset["F"] = pd.qcut(rfm_dataset["Frequency"], q = 5, labels= range(1, 6)) #[1,2,3,4,5]
rfm_dataset["M"] = pd.qcut(rfm_dataset["Frequency"], q = 5, labels= range(1, 6)) #[1,2,3,4,5]


# In[183]:


rfm_dataset


# In[184]:


# คำนวณ RFM_Score
# concept คือ การเอา R, F, M มาต่อกันเป็น string

# rfm_dataset["RFM_Group"] = rfm_dataset["R"].astype(str) + rfm_dataset["F"].astype(str) + rfm_dataset["M"].astype(str)

rfm_dataset["RFM_Score"] = rfm_dataset[["R", "F", "M"]].apply(lambda y : "".join(y.astype(str)), axis = 1)


# In[185]:


rfm_dataset


# # RFMScore

# In[186]:


# หา RFM Score 
# concept : เอา R F M มาบวกกัน

# rfm_dataset["RFM_Score"] = rfm_dataset[["R", "F", "M"]].sum(axis = 1)


# -  ใน Pandas sum(axis=1) หมายถึงการหาผลรวมของแต่ละแถว (row) ใน DataFrame
# 
#    โดยที่:
# 
# -  axis=0 (หรือไม่ระบุ axis) จะหมายถึงการหาผลรวมตามแนวแกน columns (แนวตั้ง) ดังนั้น df.sum(axis=0) จะคำนวณผลรวมของแต่ละคอลัมน์
#  - axis=1 จะหมายถึงการหาผลรวมตามแนวแกน rows (แนวนอน) ดังนั้น df.sum(axis=1) จะคำนวณผลรวมของแต่ละแถว

# In[187]:


rfm_dataset


# # 7. Mapping ข้อมูล 

#     Segment Description  
#   
# - Champions Bought recently, buy often and spend the most
# - Loyal Customers Buy on a regular basis. Responsive to promotions.
# - Potential Loyalist Recent customers with average frequency.
# - Recent Customers Bought most recently, but not often.
# - Promising Recent shoppers, but haven’t spent much.
# - Customers Needing Attention Above average recency, frequency and monetary values. May not have bought very recently though.
# - About To Sleep Below average recency and frequency. Will lose them if not reactivated.
# - At Risk Purchased often but a long time ago. Need to bring them back!
# - Can’t Lose Them Used to purchase frequently but haven’t returned for a long time.
# - Hibernating Last purchase was long back and low number of orders. May be lost.
# - Lost Lowest recency, frequency, and monetary score
# 

# In[188]:


# สร้าง column RF_Score
rfm_dataset["RF_Score"] = rfm_dataset[["R", "F"]].apply(lambda y : "".join(y.astype(str)), axis = 1)


# In[189]:


rfm_dataset


# In[190]:


segments_map = {
        r'5[4-5]': 'Champions',
        r'[3-4][4-5]': 'Loyal Customers',
        r'[4-5][2-3]': 'Potential Loyalist',
        r'51': 'New Customers',
        r'41': 'Promising',
        r'33': 'Need Attention',
        r'3[1-2]': 'About To Sleep',
        r'[1-2][3-4]': 'At Risk',
        r'[1-2]5': 'Can’t Lose Them',
        r'[1-2][1-2]': 'Hibernating'
          }


# In[192]:


rfm_dataset


# In[194]:


#แทนค่า
rfm_dataset["Segment"] = rfm_dataset["RF_Score"].replace(segments_map, regex = True)


# In[195]:


rfm_dataset


# In[196]:


#save ออกไป
rfm_dataset.to_csv("1.RFM_ Customer_Segmentation .csv")


# # 1. ลูกค้าแต่ละ segmentation มีเท่าไหร่

# In[197]:


rfm_dataset.shape


# In[198]:


rfm_dataset["Segment"].value_counts()


# # 2. Check segments mean, min and max

# In[199]:


rfm_dataset.groupby(["Segment"])[['Recency', 'Frequency', 'Monetary']].agg(['mean', 'min', 'max']).sort_values(by=[('Recency', 'mean')], ascending=True)
                                                                                                            


# # 3. Who are the best customers (Champions)?

# In[200]:


rfm_dataset[ rfm_dataset["Segment"] == "Champions"]


#       Champions:  Customer Behavior
# - กลุ่มลูกค้าที่เพิ่งเข้ามาซื้อสินค้าเมื่อไม่นานมานี้
# - มีการมาซื้อเป็นประจำ
# - ยอดซื้อสูง
# 
#       Champions: Customer Relationship Management
# - ออกโปรโมชั่นในกลุ่มสินค้าที่ลูกค้ากลุ่มนี้ยังไม่เคยซื้อและมีแนวโน้มว่าน่าจะซื้อ
# - ควรขยายโปรโมชั่นไปที่กลุ่มสินค้าอื่นด้วย 
# - หรือแจกรางวัล สิทธิพิเศษใหม่ ๆ จากการสะสมแต้ม เพื่อดึงดูดให้ลูกค้ากลับมาซื้อของซ้ำ ๆ 

# # 4. Loyal Customers 

# In[201]:


rfm_dataset[ rfm_dataset["Segment"] == "Loyal Customers"]


#       Loyal Customers: Customer Behavior
# - กลุ่มลูกค้าที่เพิ่งเข้ามาซื้อสินค้าเมื่อไม่นานมานี้
# - มีการมาซื้อค่อนข้างบ่อย
# - ยอดซื้อสูง
# 
#       Loyal Customers: Customer Relationship Management
# - ควรนำเสนอความแปลกใหม่ 
# - ส่งข้อความเพื่ออัปเดตสินค้าใหม่ 
# - ส่งโปรโมชั่นใหม่ ที่เป็นสินค้าที่มีความใกล้เคียงกับสิ่งที่ลูกค้าซื้อบ่อย
# - เพิ่มความสัมพันธ์ให้ลูกค้ารักแบรนด์มากขึ้น ด้วยการมอบของขวัญสุดพิเศษให้กับลูกค้า เช่น ส่วนลดพิเศษ ของขวัญ หรือ Voucher

# # 5. Potential Loyalist

# In[202]:


rfm_dataset[ rfm_dataset["Segment"] == "Potential Loyalist"]


#       Potential Loyalist : Customer Behavior
# - กลุ่มลูกค้าที่เพิ่งเข้ามาซื้อสินค้าเมื่อไม่นานมานี้
# - ความถี่ในการมาซื้ออยู่ในระดับปานกลาง
# - ยอดซื้อค่อนข้างสูง
# 
#       Potential Loyalist : Customer Relationship Management
# - เป็นกลุ่มลูกค้าที่มีแนวโน้มจะมาเป็นลูกค้าขาประจำ
# - ควรเน้นให้ลูกค้ามาซื้อสินค้าให้บ่อยมากยิ่งขึ้น ด้วยเทคนิคการทำโปรโมชั่นที่น่าดึงดูด
# - เช่น ส่งบรอดแคสต์ข้อความ เพื่อบอกโปรโมชั่นที่กำหนดเวลาหมดอายุ เพื่อกระตุ้นให้ลูกค้ากลับมาซื้อสินค้าภายในระยะเวลาที่กำหนด
# - แจกของรางวัลจากการแลกพอยท์ เพื่อสร้างการรับรู้ว่า พอยท์ที่ลูกค้ามีสามารถแลกของได้หลายอย่าง

# # 6. New Customers:

# In[203]:


rfm_dataset[( rfm_dataset["Segment"] == "New Customers")]


#         New Customers: Customer Behavior
# - กลุ่มลูกค้าใหม่ ที่เพิ่งเริ่มมาซื้อสินค้าเมื่อไม่นานมานี้
# - ความถี่ในการมาซื้ออยู่ในระดับปานกลาง
# - ยอดซื้อค่อนข้างสูง
# 
#         New Customers: Customer Relationship Management
# - กลุ่มนี้ถือว่าเป็นกลุ่มลูกค้าที่เราต้องเน้นให้เขากลับมาซื้อของเราในครั้งถัดไปให้ได้ 
# - เพื่อขยับจากสถานะลูกค้าใหม่ไปเป็นลูกค้าประจำ 
# - ด้วยเทคนิคกระตุ้นให้เกิดการซื้อซ้ำ เช่น Up-Sell หรือ Cross-sell(การจับคู่สินค้า)

# # 7. Promising

# In[204]:


rfm_dataset[( rfm_dataset["Segment"] == "Promising")]


#       Promising: Customer Behavior
# - กลุ่มลูกค้าที่เพิ่งเข้ามาซื้อสินค้าเมื่อไม่นานมานี้
# - ไม่ค่อยมาซื้อบ่อย
# - ยอดซื้อน้อยมาก
# 
#       Promising: Customer Relationship Management
# - กลุ่มนี้ต้องกระตุ้นให้กลับมาซื้อสินค้าอีกครั้งให้ได้ก่อน 
# - และขั้นต่อไป คือ ทำให้ลูกค้ากลับซื้อสินค้าถี่มากยิ่งขึ้น 
# - ออกโปรโมชั่นมีความน่าดึงดูด แสดงถึงความคุ้มค่าจริง ๆ เพื่อลูกค้ารู้สึกว่าที่หาไม่ได้จากที่ไหนแล้ว

# # 8. Need Attention

# In[205]:


rfm_dataset[( rfm_dataset["Segment"] == "Need Attention")]


#       Need Attention: Customer Behavior
# - กลุ่มลูกค้าเคยมาซื้อ แล้วห่างหายไป
# - ไม่ค่อยมาซื้อบ่อย
# - ยอดซื้อปานกลาง
# 
#       Need Attention: Customer Relationship Management
# - เป็นกลุ่มที่จะไม่กลับมาซื้อซ้ำและอาจห่างหายไปในที่สุด
# - กระตุ้นด้วยโปรโมชั่นที่ถ้าไม่รีบมาซื้อตอนนี้อาจจะพลาดสิ่งที่ดีที่สุดไป
# - โปรโมชั่นนั้นๆ จะนำเสนอสินค้าที่ลูกค้าสนใจ โดยวิเคราะห์ได้จากข้อมูลสินค้าที่ลูกค้าเคยซื้อครั้งก่อน
# - หรือ คัดเลือกกลุ่มสินค้าอื่น ๆ ที่มีความใกล้เคียงกัน ก็จะสร้างความแปลกใหม่ แต่ยังอยู่บนความสนใจของลูกค้า

# # 9. About to Sleep

# In[206]:


rfm_dataset[( rfm_dataset["Segment"] == "About To Sleep")]


#      About to Sleep: Customer Behavior
# - เริ่มไม่กลับมาใช้บริการสักระยะหนึ่งแล้ว
# - ซื้อไม่บ่อย
# - ยอดซื้อปานกลาง
# 
#       About to Sleep: Customer Relationship Management
# - ด้วยความที่ลูกค้าเคยมาซื้อสินค้าแต่หายไป
# - อาจเพราะไม่ค่อยเห็นการอัปเดตสินค้า หรือยังไม่มีสินค้าอะไรที่น่าสนใจในช่วงนี้
# - อัปเดตสินค้าใหม่ๆ แจกคูปองส่วนลดแบบพิเศษในโอการต่างๆ เช่น วันเกิด วันเทศกาลต่างๆ
# - ให้ลูกค้ารู้สึกได้รับความพิเศษ และกำหนดช่วงระยะเวลาในการใช้คูปอง เพื่อให้ลูกค้ากลับมาซื้อสินค้าอย่างรวดเร็ว

# # 10. At Risk 

# In[207]:


rfm_dataset[( rfm_dataset["Segment"] == "At Risk")]


#       At Risk: Customer Behavior
# - ไม่กลับมาใช้บริการสักระยะหนึ่งแล้ว
# - ความถี่ในการมาซื้ออยู่ในระดับปานกลาง
# - ยอดซื้อปานกลาง
# 
#       At Risk: Customer Relationship Management
# - ลูกค้ากลุ่มนี้มีโอกาสสูงที่จะหายไปเลย
# - เพราะแต่เดิมก็มีการซื้อสินค้าที่ไม่ได้สูงมาก และไม่มาซื้อสักระยะแล้ว
# - กระตุ้นการกลับมาซื้อด้วยการอัปเดตสินค้าใหม่ ๆ หรือโปรโมชั่นพิเศษ เช่น ลดราคา
# - จัดแคมเปญทดลองใช้สินค้าใหม่ฟรี เพื่อให้ลูกค้าได้เปิดใจอีกครั้ง

# # 11. Can’t Lose Them

# In[208]:


rfm_dataset[( rfm_dataset["Segment"] == "Can’t Lose Them")]


#       Can’t Lose Them: Customer Behavior
# - ไม่กลับมาใช้บริการสักระยะหนึ่งแล้ว
# - เคยมาซื้อบ่อยมาก 
# - ยอดซื้อสูง
# 
#       Can’t Lose Them: Customer Relationship Management
# - อาจจะไม่สนใจสินค้าในขณะนี้ 
# - อาจจะเปลี่ยนไปใช้สินค้าของแบรนด์อื่นแล้ว
# - ควรอัปเดตสินค้าใหม่ ที่ลูกค้ายังไม่เคยลองใช้
# - โปรโมชั่นที่แพ็กคู่ที่ให้ทั้งส่วนลดและของแถม
# 

# # 12. Hibernating

# In[209]:


rfm_dataset[( rfm_dataset["Segment"] == "Hibernating")]


#      Hibernating: Customer Behavior
# - ไม่กลับมาใช้บริการสักระยะหนึ่งแล้ว
# - มาซื้อไม่บ่อย
# - ยอดซื้อน้อย
# 
#       Hibernating: Customer Relationship Management
# - เป็นลูกค้าที่มีโอกาสกลับมาซื้อซ้ำได้ยากมาก
# - ไม่ควรกระตุ้นด้วยโปรโมชั่น หรือขายแบบ hard saleอาจจะดูเป็นยัดเยียดการขายมากเกินไปจนลูกค้าอึกอัด
# - ส่งแบบสอบถามถึงความพึงพอใจที่มีต่อสินค้า แล้วแจกของขวัญเป็นการตอบแทน
# - เพราะจะได้รู้ถึงสาเหตุที่แท้จริง เพื่อที่จะได้นำมาปรับปรุงสินค้าให้ดีขึ้นต่อไป

# In[210]:


rfm_dataset


# In[211]:


segments_counts = rfm_dataset["Segment"].value_counts().sort_values( ascending = True)
segments_counts 


# In[212]:


segments_counts.to_csv("segments_counts.csv")

