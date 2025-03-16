import streamlit as st

def show_home():
    st.title("🐱🐶 แนวทางการพัฒนาโมเดล Cat vs Dog Classifier")
    st.write("""
    โครงการนี้พัฒนาขึ้นเพื่อสร้างแบบจำลองการเรียนรู้เชิงลึก (**Deep Learning Model**) 
    สำหรับจำแนกภาพของแมวและสุนัข โดยใช้ **Convolutional Neural Network (CNN)**
    """)

    st.header("📌 ขั้นตอนการพัฒนา")
    
    st.subheader("1️⃣ การเตรียมข้อมูล")
    st.write("""โโมเดลนี้ใช้ข้อมูลภาพจาก Kaggle ซึ่งมีรูปภาพของหมาและแมว โดยแต่ละภาพมีลักษณะสำคัญดังนี้ :""")
    st.write("""
    - **Feature หลัก :** ขนาดภาพ, สี (RGB Values), ลักษณะของใบหน้าและลำตัว
    - **ตัวแปรเป้าหมาย :** จำแนกเป็น 2 คลาส คือ หมา (Dog) และ แมว (Cat)
    """)

    st.subheader("2️⃣ ทฤษฎีของอัลกอริทึม CNN")
    st.write("""**CNN (Convolutional Neural Network)**""")
    st.write("""
    - โครงสร้างหลักประกอบด้วย Convolutional Layer, Pooling Layer, Fully Connected Layer
    - ใช้ฟังก์ชัน Activation เช่น ReLU และ Softmax
    - เหมาะกับปัญหาที่เกี่ยวกับการประมวลผลภาพ เช่น การจำแนกภาพหมาและแมว
    """)
    
    st.subheader("3️⃣ ขั้นตอนการพัฒนาโมเดล CNN")
    st.subheader("การโหลดและเตรียมข้อมูล")
    st.code("""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        # กำหนดขนาดภาพและ batch size
        image_size = (128, 128)
        batch_size = 32

        # โหลดข้อมูลภาพและทำ Data Augmentation
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_generator = train_datagen.flow_from_directory(
            'dataset/cat_dog',
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )
        validation_generator = train_datagen.flow_from_directory(
            'dataset/cat_dog',
            target_size=image_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )
    """)

    st.subheader("การสร้างและเทรนโมเดล CNN")
    st.code("""
        # สร้างโมเดล CNN
        cnn_model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        # คอมไพล์และเทรนโมเดล
        cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        cnn_model.fit(train_generator, validation_data=validation_generator, epochs=10)

        # บันทึกโมเดล
        cnn_model.save('cnn_cat_dog_classifier.h5')
    """)

    st.subheader("4️⃣ การใช้งานโมเดล CNN")
    st.code("""
        from tensorflow.keras.models import load_model
        import numpy as np
        from tensorflow.keras.preprocessing import image

        # โหลดโมเดล
        cnn_model = load_model('cnn_cat_dog_classifier.h5')

        # โหลดและประมวลผลภาพใหม่
        img = image.load_img('test_pet.jpg', target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ทำนายผล
        prediction = cnn_model.predict(img_array)
        if prediction[0][0] > 0.5:
            print("Predicted: Dog")
        else:
            print("Predicted: Cat")
    """)
 
    st.success("🎯 ระบบพร้อมใช้งานแล้ว! กรุณาเลือกเมนูจากแถบด้านซ้ายเพื่อทดลองใช้งาน")
