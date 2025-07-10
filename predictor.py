{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b8953cd4-3289-4282-89e6-ec541e2acf62",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6e246096-ee2a-432d-815e-60e427f7f324",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "27cc87ac-6c43-4c79-af4e-8727968b2680",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a560639e-cc30-4249-b52a-81f69d4aea04",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "621ba220-48a3-4e10-9249-0aabb150a584",
   "metadata": {},
   "outputs": [],
   "source": [
    "import shap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "96582831-5594-4c30-b383-9c430265d230",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "842a14fd-77ee-4876-9c3b-d78b3e1525ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = joblib.load('RF.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "78c5037f-884f-42c9-a61c-9d6f319a94c8",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = pd.read_csv('X_test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "c459487b-f951-41e4-a4d5-7a64d5079a6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#定义特征名称，对应数据集中的列名\n",
    "feature_names = [\"BC\",\"YiDC\", \"PDC\", \"Age\", \"Pension\", \"WHtR\", \"CO\", \"BMI\", \"Smoking\", \"SCL\", \"Sleepquality\", \"Pain\", \"Eyesight\", \"Diffaction\", \"Hyperlipidemia\", \"Hyperuricemia\",\"FLD\", \"OA\", \"Diabetes\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "42af3ed1-4ef5-4b07-bba1-54a807596663",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Streamlit 用户界面\n",
    "st.title(\"老年高血压预测器\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "4ec4be93-3433-467b-bad7-8746a807f0f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "BC = st.selectbox(\"平和质类型 (BC):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "2bebe670-87de-4a93-897f-16c13730df37",
   "metadata": {},
   "outputs": [],
   "source": [
    "YiDC = st.selectbox(\"阴虚质类型 (YiDC):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "45bf307c-76f7-4f9d-bc06-b3f6c66ae04f",
   "metadata": {},
   "outputs": [],
   "source": [
    "PDC = st.selectbox(\"痰湿质类型 (PDC):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "dcd57c63-b787-4ac9-a50c-836f03ddb0dc",
   "metadata": {},
   "outputs": [],
   "source": [
    "Age = st.selectbox(\"年龄 (Age):\", options=[0, 1,2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "e159454b-fe59-4db4-82fc-24e8255f08ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "Pension = st.selectbox(\"医保类型 (Pension):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "7c81279d-c7c2-4a5a-a975-1815e8b7ac5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "WHtR = st.selectbox(\"腰高比 (WHtR):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "1586ed13-41f3-4bf8-b4cd-c20eb288a4ca",
   "metadata": {},
   "outputs": [],
   "source": [
    "CO = st.selectbox(\"中心性肥胖 (CO):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "e4ccd82b-4218-45ee-859f-1c859d9a9164",
   "metadata": {},
   "outputs": [],
   "source": [
    "BMI = st.selectbox(\"体质指数 (BMI):\", options=[0,1,2,3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "8d94fcaa-bdba-4edf-ab0a-9e8606f82c33",
   "metadata": {},
   "outputs": [],
   "source": [
    "Smoking = st.selectbox(\"吸烟 (Smoking):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "83633f6c-cf8b-4fb3-947a-40578f4f9a2d",
   "metadata": {},
   "outputs": [],
   "source": [
    "SCL = st.selectbox(\"精神文化生活 (SCL):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "f608ce2c-f924-461c-9f4a-782d2b9fc281",
   "metadata": {},
   "outputs": [],
   "source": [
    "Sleepquality = st.selectbox(\"睡眠质量 (Sleepquality):\", options=[0,1,2,3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "b53295f7-5046-4bee-9cfc-fc2d1f9b552e",
   "metadata": {},
   "outputs": [],
   "source": [
    "Pain = st.selectbox(\"身体疼痛 (Pain):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "b539fd41-19f0-4f0e-8009-ac40c9239371",
   "metadata": {},
   "outputs": [],
   "source": [
    "Eyesight = st.selectbox(\"视力 (Eyesight):\", options=[0, 1,2,3])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "9a57fdf2-d980-4c5d-8d70-9ea027269ce1",
   "metadata": {},
   "outputs": [],
   "source": [
    "Diffaction = st.selectbox(\"行动困难 (Diffaction):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "9b3eef7a-2412-410b-ae8a-47a9f2fdf0ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "Hyperlipidemia = st.selectbox(\"高脂血症 (Hyperlipidemia):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "86cabed9-8a86-41ba-ab85-95e75f1c7f85",
   "metadata": {},
   "outputs": [],
   "source": [
    "Hyperuricemia = st.selectbox(\"高尿酸血症 (Hyperuricemia):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "0fca68b0-d0d9-4c19-bed3-e4bbc80d5f08",
   "metadata": {},
   "outputs": [],
   "source": [
    "FLD = st.selectbox(\"脂肪肝 (FLD):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "db5c03f4-e649-427e-ab13-31d15ef5eff1",
   "metadata": {},
   "outputs": [],
   "source": [
    "OA = st.selectbox(\"关节炎 (OA):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "afe34642-ca0a-49c2-bd5f-07d99f1aba0e",
   "metadata": {},
   "outputs": [],
   "source": [
    "Diabetes = st.selectbox(\"糖尿病 (Diabetes):\", options=[0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "4dd4ae2e-d9a7-476b-9e60-d01061fbf9b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 实现输入数据并进行预测\n",
    "feature_values = [BC,YiDC, PDC, Age, Pension, WHtR, CO, BMI, Smoking, SCL, Sleepquality, Pain, Eyesight, Diffaction, Hyperlipidemia, Hyperuricemia,FLD, OA, Diabetes]  # 将用户输入的特征值存入列表\n",
    "features = np.array([feature_values])  # 将特征转换为 NumPy 数组，适用于模型输入"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "id": "c03b8d32-91dd-4dbe-8704-4ff16ff9a8e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 当用户点击 \"Predict\" 按钮时执行以下代码\n",
    "if st.button(\"Predict\"):\n",
    "    # 预测类别（0: 无高血压，1: 有高血压）\n",
    "    predicted_class = model.predict(features)[0]\n",
    "    # 预测类别的概率\n",
    "    predicted_proba = model.predict_proba(features)[0]\n",
    "\n",
    "    # 显示预测结果\n",
    "    st.write(f\"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)\")\n",
    "    st.write(f\"**Prediction Probabilities:** {predicted_proba}\")\n",
    "\n",
    "    # 根据预测结果生成建议\n",
    "    probability = predicted_proba[predicted_class] * 100\n",
    "    # 如果预测类别为 1（高风险）\n",
    "    if predicted_class == 1:\n",
    "        advice = (\n",
    "            f\"According to our model, you have a high risk of hypertension. \"\n",
    "            f\"The model predicts that your probability of having hypertension is {probability:.1f}%. \"\n",
    "            \"It's advised to consult with your healthcare provider for further evaluation and possible intervention.\"\n",
    "        )\n",
    "   # 如果预测类别为 0（低风险）\n",
    "    else:\n",
    "        advice = (\n",
    "            f\"According to our model, you have a low risk of hypertension. \"\n",
    "            f\"The model predicts that your probability of not having hypertension is {probability:.1f}%. \"\n",
    "            \"However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider.\"\n",
    "        )\n",
    "    st.write(advice)\n",
    "    # SHAP 解释\n",
    "    st.subheader(\"SHAP Force Plot Explanation\")\n",
    "    # 创建 SHAP 解释器，基于树模型（如随机森林）\n",
    "    explainer_shap = shap.TreeExplainer(model)\n",
    "    # 计算 SHAP 值，用于解释模型的预测\n",
    "    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))\n",
    "\n",
    "    # 根据预测类别显示 SHAP 强制图\n",
    "    # 期望值（基线值）\n",
    "    # 解释类别 1（患病）的 SHAP 值\n",
    "    # 特征值数据\n",
    "    # 使用 Matplotlib 绘图\n",
    "    if predicted_class == 1:\n",
    "        shap.force_plot(explainer_shap.expected_value[1], shap_values[:, :, 1], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)\n",
    "    # 期望值（基线值）\n",
    "    # 解释类别 0（未患病）的 SHAP 值\n",
    "    # 特征值数据\n",
    "    # 使用 Matplotlib 绘图\n",
    "    else:\n",
    "         shap.force_plot(explainer_shap.expected_value[0], shap_values[:, :, 0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)\n",
    "\n",
    "    plt.savefig(\"shap_force_plot.png\", bbox_inches='tight', dpi=1200)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "818ccbb6-6df5-4393-a895-a63fc17b1b47",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
