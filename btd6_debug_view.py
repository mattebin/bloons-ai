import cv2

# Boxes are (x1,y1,x2,y2) in 1280x720 coordinates (starter guesses from your screenshot)
boxes = [
    [131, 58, 176, 81],   # LIVES
    [270, 62, 520, 83],   # MONEY (lane, widened)
    [928, 65, 1014, 87],  # ROUND
]

names  = ["LIVES", "MONEY", "ROUND"]
colors = [(0,255,0), (255,0,0), (0,0,255)]

selected = 0
step = 5

def clamp_box(b, w, h):
    x1,y1,x2,y2 = b
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    b[:] = [x1,y1,x2,y2]

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("OBS Virtual Camera not available (index 0)")

print("Controls: 1/2/3 select box | arrows move | [ ] resize | -/= step | p print | q quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # draw boxes
    for i, b in enumerate(boxes):
        clamp_box(b, w, h)
        x1,y1,x2,y2 = b
        cv2.rectangle(frame, (x1,y1), (x2,y2), colors[i], 2)
        label = f"{i+1}:{names[i]}"
        if i == selected:
            label += f"  (selected, step={step})"
        cv2.putText(frame, label, (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

    cv2.imshow("BTD6 HUD calibrator (OBS Virtual Cam)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('1'), ord('2'), ord('3')):
        selected = key - ord('1')
    elif key == ord('p'):
        print("\nCurrent boxes (x1,y1,x2,y2):")
        for i, b in enumerate(boxes):
            print(f"{names[i]}: {tuple(b)}")
    elif key == ord('-'):
        step = max(1, step - 1)
    elif key == ord('='):
        step = min(50, step + 1)
    elif key == ord('['):  # shrink
        b = boxes[selected]
        b[0] += step; b[1] += step; b[2] -= step; b[3] -= step
    elif key == ord(']'):  # grow
        b = boxes[selected]
        b[0] -= step; b[1] -= step; b[2] += step; b[3] += step

    # Arrow keys come through as 81/82/83/84 in OpenCV on many systems
    if key == 81:   # left
        boxes[selected][0] -= step; boxes[selected][2] -= step
    elif key == 83: # right
        boxes[selected][0] += step; boxes[selected][2] += step
    elif key == 82: # up
        boxes[selected][1] -= step; boxes[selected][3] -= step
    elif key == 84: # down
        boxes[selected][1] += step; boxes[selected][3] += step

cap.release()
cv2.destroyAllWindows()

