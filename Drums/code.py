import cv2
import mediapipe as mp
import pygame

# Initialize Pygame for sound
pygame.init()
sounds = {
    'snare': pygame.mixer.Sound('snare.wav'),
    'bass': pygame.mixer.Sound('bass.wav'),
    'hihat': pygame.mixer.Sound('hithat.wav')
}

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define screen zones for each drum element

WIDTH, HEIGHT = 640, 480
drum_zones = {
    'snare': (WIDTH // 4, HEIGHT // 2),     # left side for snare
    'bass': (WIDTH // 2, HEIGHT // 2),      # center for bass
    'hihat': (3 * WIDTH // 4, HEIGHT // 2)  # right side for hi-hat
}
zone_radius = 50  # Radius around the zone center to trigger sound

# Set up webcam
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        print("Error: Camera feed not available.")
        break

    # Flip and convert to RGB for MediaPipe processing
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # Draw drum zones on screen
    for drum, (x, y) in drum_zones.items():
        color = (255, 0, 0) if drum == 'snare' else (0, 255, 0) if drum == 'bass' else (0, 0, 255)
        cv2.circle(img, (x, y), zone_radius, color, 2)
        cv2.putText(img, drum.upper(), (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Process hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get index finger tip position (landmark 8)
            x = int(hand_landmarks.landmark[8].x * WIDTH)
            y = int(hand_landmarks.landmark[8].y * HEIGHT)

            # Check if the finger is in any drum zone
            for drum, (zone_x, zone_y) in drum_zones.items():
                distance = ((x - zone_x) ** 2 + (y - zone_y) ** 2) ** 0.5
                if distance < zone_radius:
                    sounds[drum].play()  # Play corresponding sound
                    cv2.circle(img, (x, y), 10, (0, 255, 255), -1)  # Highlight fingertip when hitting a drum
                    break

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the camera feed with overlay
    cv2.imshow("Virtual Drum Set", img)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.quit()