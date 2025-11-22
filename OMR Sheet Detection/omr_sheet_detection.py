import streamlit as st
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image



def process_omr(image, answer_key):
    img = np.array(image)
    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    big = []
    for c in contours:
        if cv2.contourArea(c) > 800:
            big.append(c)

    sorted_area = sorted(big, key=cv2.contourArea, reverse=False)
    # bubble block
    max_box = None
    max_size = 0

    for c in sorted_area:
        x, y, w, h = cv2.boundingRect(c)
        s = x + y + w + h
        if s > max_size:
            max_size = s
            max_box = (x, y, w, h)

    if max_box is None:
        return None, None, None, None

    x, y, w, h = max_box
    bubble_block = original[y:y+h, x:x+w]
    bubble_block = bubble_block[:, 65:]

    # Split rows
    def split_rows(img, n):
        rows = []
        h = img.shape[0]
        rh = h // n
        for i in range(n):
            rows.append(img[i*rh:(i+1)*rh, :])
        return rows

    def split_cols(img, n):
        cols = []
        w = img.shape[1]
        cw = w // n
        for i in range(n):
            cols.append(img[:, i*cw:(i+1)*cw])
        return cols

    def is_filled(opt):
        g = cv2.cvtColor(opt, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY)
        return cv2.countNonZero(b)

    rows = split_rows(bubble_block, 10)

    marked = []
    not_attempted = 0

    for row in rows:
        cols = split_cols(row, 4)
        arr = [is_filled(c) for c in cols]
        mn = min(arr)
        idx = arr.index(mn)

        count_min = sum([1 for v in arr if v == mn])
        if count_min == 4:   # no bubble filled
            marked.append(0)
            not_attempted += 1
        else:
            marked.append(idx+1)

    # Draw output
    out = bubble_block.copy()
    rh = out.shape[0] // 10
    cw = out.shape[1] // 8

    for i in range(10):
        ans = marked[i]
        if ans != 0:
            cx = (ans * rh) + rh//4
            cy = (i * cw) + cw//4
            cv2.circle(out, (cx+50, cy+20), 10, (255, 0, 0), 2)

    for i in range(10):
        ans = answer_key[i]
        if marked[i] != 0:
            cx = (ans * rh) + rh//4
            cy = (i * cw) + cw//4
            cv2.circle(out, (cx+50, cy+20), 10, (0, 255, 0), 2)

    correct = sum([1 for i in range(10) if marked[i] == answer_key[i]])
    score = correct * 10

    return marked, correct, score, out


# ============================
# STREAMLIT UI
# ============================

st.title("📘 OMR Sheet Checker")

if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["Name", "Subject", "Correct", "Score"])

answer_key = [1,2,3,4,4,3,2,1,2,2]

uploaded = st.file_uploader("Upload OMR Sheet", type=['png', 'jpg', 'jpeg'])

student_name = st.text_input("Student Name")
subject = st.text_input("Subject")

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded OMR", width=400)

    if st.button("Process OMR"):
        marked, correct, score, checked_img = process_omr(img, answer_key)

        if marked is None:
            st.error("Could not detect OMR bubbles. Try clearer sheet.")
        else:
            st.success(f"Correct Answers: {correct}")
            st.success(f"Score: {score} / 100")

            st.write("### Checked OMR Sheet")
            st.image(checked_img, channels="BGR", width=400)

            # Add to DataFrame
            new_row = {
                "Name": student_name,
                "Subject": subject,
                "Correct": correct,
                "Score": score
            }
            st.session_state.results_df = pd.concat(
                [st.session_state.results_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

st.write("### Results DataFrame")
st.dataframe(st.session_state.results_df)

# Download CSV
csv = st.session_state.results_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")

