import os
import cv2
import dlib
import numpy as np
from tkinter import Tk, Label, Button, Entry, messagebox, simpledialog, Toplevel
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sqlite3
from datetime import datetime, timedelta
import csv


last_identification_time = datetime.min
recent_access = {}

def log_access(user_name, status):
    global recent_access  # Usar el diccionario global

    # Obtener el tiempo actual
    current_time = datetime.now()

    # Verificar si ya se registró recientemente
    if user_name in recent_access:
        last_time = recent_access[user_name]
        time_diff = (current_time - last_time).total_seconds()

        # Si la diferencia es menor a 10 segundos, no registrar
        if time_diff < 10:
            return

    # Registrar el nuevo acceso
    conn = sqlite3.connect('access_log.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS access_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_name TEXT,
                        status TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute("INSERT INTO access_log (user_name, status) VALUES (?, ?)", (user_name, status))
    conn.commit()
    conn.close()

    # Actualizar el tiempo de acceso reciente
    recent_access[user_name] = current_time


# Función para capturar imágenes de un nuevo usuario
def add_new_user():
    user_name = simpledialog.askstring("Nuevo Usuario", "Introduce el nombre del usuario:")
    if not user_name:
        messagebox.showwarning("Advertencia", "El nombre del usuario no puede estar vacío.")
        return

    # Crear directorio para el nuevo usuario
    user_dir = f"dataset/{user_name}"
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    cap = cv2.VideoCapture(0)
    image_count = 0
    messagebox.showinfo("Captura", "Presiona 's' para capturar imágenes y 'q' para finalizar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Captura de Rostro", frame)
        key = cv2.waitKey(1)

        # Capturar imagen con 's'
        if key & 0xFF == ord('s'):
            image_path = os.path.join(user_dir, f"{user_name}_{image_count}.jpg")
            cv2.imwrite(image_path, frame)
            image_count += 1
            print(f"Imagen {image_count} guardada en {image_path}")

        # Finalizar captura con 'q'
        elif key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Procesar imágenes del nuevo usuario
    process_user_images(user_name)

# Función para procesar imágenes y generar embeddings
def process_user_images(user_name):
    input_dir = f"dataset/{user_name}"
    faces_dir = f"{input_dir}_faces"
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for i, (x, y, w, h) in enumerate(faces):
            face = img[y:y + h, x:x + w]
            face_path = os.path.join(faces_dir, f"{img_name.split('.')[0]}_face_{i}.jpg")
            cv2.imwrite(face_path, face)
            print(f"Rostro guardado en {face_path}")

    generate_embeddings(user_name, faces_dir)

# Función para generar embeddings del usuario
def generate_embeddings(user_name, faces_dir):
    face_detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    embeddings = []
    labels = []

    for img_name in os.listdir(faces_dir):
        img_path = os.path.join(faces_dir, img_name)
        img = dlib.load_rgb_image(img_path)
        faces = face_detector(img, 1)

        for face in faces:
            shape = shape_predictor(img, face)
            face_embedding = np.array(face_recognizer.compute_face_descriptor(img, shape))
            embeddings.append(face_embedding)
            labels.append(user_name)

    if os.path.exists('embeddings.npy') and os.path.exists('labels.npy'):
        existing_embeddings = np.load('embeddings.npy')
        existing_labels = np.load('labels.npy')
        embeddings = np.vstack((existing_embeddings, embeddings))
        labels = np.concatenate((existing_labels, labels))

    np.save('embeddings.npy', embeddings)
    np.save('labels.npy', labels)
    messagebox.showinfo("Éxito", f"Usuario {user_name} agregado con éxito.")

# Función para entrenar el modelo SVM
def train_svm():
    embeddings = np.load('embeddings.npy')
    labels = np.load('labels.npy')

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f"Precisión del modelo: {accuracy * 100:.2f}%")
    return clf

# Función de detección e identificación
def detect_and_identify():
    global last_identification_time  # Usar una variable global para controlar el tiempo
    clf = train_svm()
    if clf is None:
        return  # Salir si no hay un modelo entrenado

    # Configurar detector, reconocedor y predictor
    face_detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Cargar etiquetas y usuarios registrados
    labels = np.load('labels.npy', allow_pickle=True)
    unique_labels = np.unique(labels)

    # Rastrear usuarios detectados durante la sesión
    detected_users = set()

    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Info", "Presiona 'q' para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = datetime.now()
            if (current_time - last_identification_time).total_seconds() < 1:  # Intervalo de 3 segundos
                continue

            faces = face_detector(frame, 1)
            for face in faces:
                shape = shape_predictor(frame, face)
                face_embedding = np.array(face_recognizer.compute_face_descriptor(frame, shape))

                # Obtener la predicción y las probabilidades
                proba = clf.predict_proba([face_embedding])[0]
                prediction_idx = np.argmax(proba)
                confidence = proba[prediction_idx]

                # Verificar si la confianza es suficiente para identificar
                if confidence > 0.5:
                    label = unique_labels[prediction_idx]
                    if label not in detected_users:  # Evitar registros duplicados
                        log_access(label, "Autorizado")
                        detected_users.add(label)  # Agregar al conjunto de usuarios detectados
                        print(f"Acceso permitido: {label}")
                else:
                    label = "Desconocido"
                    log_access(label, "Denegado")
                    print("Acceso denegado")

                # Dibujar la caja y etiqueta
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Actualizar el tiempo de la última identificación
                last_identification_time = current_time

            cv2.imshow("Identificación en Tiempo Real", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"Error durante la identificación: {e}")
    finally:
        # Liberar recursos en cualquier caso
        cap.release()
        cv2.destroyAllWindows()
        print("Recursos liberados correctamente.")

        # Registrar inasistencias
        register_absences(unique_labels, detected_users)


def register_absences(registered_users, detected_users):
    """
    Compara los usuarios detectados con los registrados y marca inasistencias.
    """
    for user in registered_users:
        if user not in detected_users:
            log_access(user, "Inasistencia")
            print(f"Inasistencia registrada: {user}")

        
    
def view_access_log():
    conn = sqlite3.connect('access_log.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM access_log")
    records = cursor.fetchall()
    conn.close()

    # Crear ventana para mostrar los registros
    log_window = Toplevel()
    log_window.title("Registro de Accesos")
    for i, (id, user, status, timestamp) in enumerate(records):
        Label(log_window, text=f"{id} - {user} - {status} - {timestamp}").grid(row=i, column=0)


# Función para limpiar los registros en la base de datos
def clear_access_log():
    conn = sqlite3.connect('access_log.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM access_log")
    conn.commit()
    conn.close()
    messagebox.showinfo("Éxito", "Todos los registros han sido eliminados.")
    print("Registros limpiados con éxito.")

# Función para exportar los registros a un archivo CSV
def export_to_csv():
    conn = sqlite3.connect('access_log.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM access_log")
    records = cursor.fetchall()
    conn.close()

    if not records:
        messagebox.showinfo("Info", "No hay registros para exportar.")
        return

    # Crear la carpeta si no existe
    export_dir = "registros_csv"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    # Obtener la fecha actual y construir la ruta del archivo
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"asistencia-{today}.csv"
    filepath = os.path.join(export_dir, filename)

    # Guardar los registros en el archivo CSV
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Usuario", "Estado", "Fecha y Hora"])  # Encabezados
        writer.writerows(records)

    messagebox.showinfo("Éxito", f"Registros exportados a {filepath}.")
    print(f"Registros exportados a {filepath}.")

# Actualización de la interfaz gráfica principal
def main():
    root = Tk()
    root.title("Sistema de Reconocimiento Facial")
    root.geometry("400x400")

    Label(root, text="Sistema de Reconocimiento Facial", font=("Arial", 16)).pack(pady=10)
    Button(root, text="Iniciar Identificación", command=detect_and_identify).pack(pady=10)
    Button(root, text="Agregar Nuevo Usuario", command=add_new_user).pack(pady=10)
    Button(root, text="Ver Registro de Accesos", command=view_access_log).pack(pady=10)
    Button(root, text="Limpiar Registros", command=clear_access_log).pack(pady=10)
    Button(root, text="Exportar a CSV", command=export_to_csv).pack(pady=10)
    Button(root, text="Salir", command=root.destroy).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
