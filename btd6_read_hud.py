import re
import cv2
import pytesseract

# ==========================
# HUD COORDINATES (1280x720)
# Format: (x1, y1, x2, y2)
# (+5 px on each side applied)
# ==========================
LIVES = (126, 53, 181, 86)
MONEY = (265, 57, 393, 88)
ROUND = (923, 60, 1019, 92)

# ==========================
# Last-good smoothing
# ==========================
last_lives = "?"
last_money = "?"
last_round = "?"

def crop(frame, box):
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]

def prep(img_bgr):
    # Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Upscale (important for HUD text)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Black-hat to emphasize dark outlines around white text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.add(gray, bh)

    # Slight blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Threshold (invert works best for BTD6 HUD)
    _, th = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return th

def ocr_best(th, whitelist):
    cfg = f"--oem 1 --psm 7 -c tessedit_char_whitelist={whitelist}"
    a = pytesseract.image_to_string(th, config=cfg).strip()
    b = pytesseract.image_to_string(255 - th, config=cfg).strip()

    def score(s):
        return sum(c.isdigit() or c == "/" for c in s)

    return a if score(a) >= score(b) else b

def draw_boxes(frame):
    cv2.rectangle(frame, LIVES[:2], LIVES[2:], (0, 255, 0), 2)
    cv2.rectangle(frame, MONEY[:2], MONEY[2:], (255, 0, 0), 2)
    cv2.rectangle(frame, ROUND[:2], ROUND[2:], (0, 0, 255), 2)

def main():
    global last_lives, last_money, last_round

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("OBS Virtual Camera not available")

    print("Reading BTD6 HUD (press q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop & preprocess
        lives_th = prep(crop(frame, LIVES))
        money_th = prep(crop(frame, MONEY))
        round_th = prep(crop(frame, ROUND))

        # OCR
        lives_txt = ocr_best(lives_th, "0123456789")
        money_txt = ocr_best(money_th, "0123456789")
        round_txt = ocr_best(round_th, "0123456789/")

        # Parse / clean
        lives_num = re.sub(r"\D", "", lives_txt)
        money_num = re.sub(r"\D", "", money_txt)

        m = re.search(r"(\d+)\s*/\s*(\d+)", round_txt)
        round_clean = f"{m.group(1)}/{m.group(2)}" if m else ""

        # Debounce rounds
        if round_clean:
            try:
                cur, total = round_clean.split("/")
                cur_i = int(cur)
                total_i = int(total)
                if 1 <= cur_i <= total_i <= 200:
                    last_round = f"{cur_i}/{total_i}"
            except Exception:
                pass

        # Last-good smoothing
        if lives_num:
            last_lives = lives_num
        if money_num:
            last_money = money_num

        # Debug overlay
        draw_boxes(frame)
        cv2.putText(
            frame,
            f"Lives={last_lives}  Money={last_money}  Round={last_round}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("BTD6 HUD OCR", frame)
        print(
            f"\rLives={last_lives}  Money={last_money}  Round={last_round}    ",
            end=""
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone.")

if __name__ == "__main__":
    main()


# ==========================
# Reusable HUD reader (for import)
# ==========================
def read_hud_from_frame(frame):
    """
    Returns a dict with best-known values using the same smoothing as the live OCR script.
    Values:
      lives: int | None
      money: int | None
      round_cur: int | None
      round_total: int | None
      round_str: str (e.g. "31/80" or "")
    """
    global last_lives, last_money, last_round

    # preprocess
    lives_th = prep(crop(frame, LIVES))
    money_th = prep(crop(frame, MONEY))
    round_th = prep(crop(frame, ROUND))

    # OCR
    lives_txt = ocr_best(lives_th, "0123456789")
    money_txt = ocr_best(money_th, "0123456789")
    round_txt = ocr_best(round_th, "0123456789/")

    # Parse / clean
    lives_num = re.sub(r"\D", "", lives_txt)
    money_num = re.sub(r"\D", "", money_txt)

    m = re.search(r"(\d+)\s*/\s*(\d+)", round_txt)
    round_clean = f"{m.group(1)}/{m.group(2)}" if m else ""

    # Debounce rounds (same rules as main())
    if round_clean:
        try:
            cur, total = round_clean.split("/")
            cur_i = int(cur)
            total_i = int(total)
            if 1 <= cur_i <= total_i <= 200:
                last_round = f"{cur_i}/{total_i}"
        except Exception:
            pass

    # Last-good smoothing
    if lives_num:
        last_lives = lives_num
    if money_num:
        last_money = money_num

    # Build return (prefer parsed ints, fall back to last_good)
    lives_i = int(last_lives) if str(last_lives).isdigit() else 100
    money_i = int(last_money) if str(last_money).isdigit() else None

    rc = rt = None
    rs = last_round if isinstance(last_round, str) else ""
    if rs and "/" in rs:
        try:
            a, b = rs.split("/")
            rc = int(a)
            rt = int(b)
        except Exception:
            rc = rt = None

    return {
        "lives": lives_i,
        "money": money_i,
        "round_cur": rc,
        "round_total": rt,
        "round_str": rs,
    }
