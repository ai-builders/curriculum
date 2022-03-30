# curriculum
8-week curriculum for AI Builders

<img src="https://github.com/ai-builders/curriculum/raw/main/images/curriculum.png" width="500">

## Week 1 - [บทที่ 1 - Machine Learning คืออะไร](https://github.com/ai-builders/curriculum/blob/main/notebooks/01_ml_what.ipynb)

ในบทเรียนนี้เราจะเรียนรู้ว่า Artificial Intelligence (AI), Machine Learning (ML) และ Deep Learning (DL) คืออะไร เหมือนกันหรือแตกต่างกันอย่างไร เราจะเรียนรู้ส่วนประกอบของระบบ machine learning และวิธีการเทรน machine learning model ด้วยตัวอย่างจำแนกรูปภาพอาหารไทย 48 ชนิดจากชุดข้อมูล [FoodyDudy](https://github.com/GemmyTheGeek/FoodyDudy) หลังจากนั้นเราจะเห็นว่าส่วนประกอบและวิธีการเทรนนี้ถูกใช้กับข้อมูลชนิดอื่นๆ เช่น ข้อความ (texts) และตาราง (tabular data) ได้อย่างไรบ้าง

บทเรียนนี้ปรับแต่งและเพิ่มเติมจาก [fastai Practical Deep Learning for Coders v4 part1 - Lesson 1](https://course.fast.ai/videos/?lesson=1) เพื่อให้เหมาะกับโครงการ AI Builders

## Week 2 - [บทที่ 2 - ชุดข้อมูลมหัศจรรย์และถิ่นที่อยู่](https://github.com/ai-builders/curriculum/blob/main/notebooks/02_fantastic_datasets.ipynb)

ในปัจจุบันชุดข้อมูลที่มีพร้อมทั้งปริมาณและคุณภาพเป็นส่วนสำคัญในการสร้าง ML models ในบทเรียนนี้เราจะเรียนรู้วิธีการหาข้อมูลมาเทรนโมเดลของเราทั้งจากชุดข้อมูล open data, web scraping, หรือสร้างขึ้นมาเองจากโมเดลและโค้ด open source ทั้งนี้การหาข้อมูลมาเทรนโมเดลจากแหล่งข้อมูลสาธารณะที่กล่าวมานั้นเราต้องให้ความสำคัญเรื่องลิขสิทธิ์และจริยธรรม (แม้แต่โมเดลเองก็สร้างข้อมูลที่ผิดลิขสิทธิ์-จริยธรรมได้; เรียนเพิ่มเติมในบทที่ 7)

## Week 3 - [บทที่ 3 - Stochastic Gradient Descent ตั้งแต่เริ่มต้น](https://github.com/ai-builders/curriculum/blob/main/notebooks/03_sgd_from_scratch.ipynb)

ในบทเรียนนี้ เราจะทำการสร้างวิธีที่โมเดลของเราเรียนรู้ในบทเรียนที่แล้วๆมา เรียกว่า stochastic gradient descent ขึ้นมาเองตั้งแต่ต้นโดยใช้เพียงแค่ Pytorch สำหรับ linear algebra และการทำ partial derivatives เท่านั้น ด้วยตัวอย่างการจำแนกรูปภาพตัวเลข 3 และ 7 ออกจากกัน

บทเรียนแปล-สรุปมาจาก [04_mnist_basics.ipynb](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb) ของ [fastai](https://course.fast.ai/) ผู้ที่สนใจสามารถไปติดตามบทเรียนต้นทางได้ที่ [course.fast.ai](https://course.fast.ai/)


## Track - Images

### Week 4 - 4i

Loren ipsum

### Week 5 - 5i

Loren ipsum

### Week 6 - 6i

Loren ipsum

## Track - Texts

### Week 4 - [บทที่ 4n - NLP คืออะไร? บทเรียนจากอดีตสู่ปัจจุบัน](https://docs.google.com/presentation/d/1jXdjU0LqgIzpbfqy8pqI8-zJUc16_0EY8iiAgxcHMyc/edit?usp=sharing)

ในบทนี้เราจะเรียนเกี่ยวกับ NLP ตั้งแต่พื้นฐาน ไปจนถึง NLP ในยุคปัจจุบันว่ามีการพัฒนาไปอย่างไรบ้าง พร้อมทั้งยกตัวอย่างการทำ text classification (การจำแนกข้อความ) ด้วยวิธีตั้งแต่อดีตยันปัจจุบัน
Notebook: [HERE](notebooks/04n_NLP_Thai_Text_Classification.ipynb)

### Week 5 - [บทที่ 5n - การเทรนโมเดลบน Hugging Face พาร์ท 1](https://docs.google.com/presentation/d/1KFzS5iL4pi-XV_QxtP9G0sxSSy0PqXvuKsDLSPDhZtA/edit?usp=sharing)

ในปัจจุบันการทำงานด้าน NLP มักจะนิยมใช้งาน Deep Learning ในการแก้ปัญหาโจทย์ที่มีความซับซ้อนสูง โดย Library ที่เป็นที่นิยมในปัจจุบันคือ "Hugging Face" โดยในบทเรียนนี้เราจะมาเรียนการใช้งาน Hugging Face เพื่อเทรนโมเดลในงานด้าน NLP !
Notebook: [Part1](), [Part2](), [Part3]()

### Week 6 - [บทที่ 6n - การเทรนโมเดลบน Hugging Face พาร์ท 2](https://docs.google.com/presentation/d/1lpnF9ZteWukd5RVBmf1mN9ZKaUoxhNByD0rTAeEqp00/edit?usp=sharing)

หลังจากที่เราได้เรียนรู้พื้นฐานของการใช้งาน Hugging Face แล้ว เราจะมาเทรนโมเดลเพื่องานที่ซับซ้อนมากขึ้น เช่น Machine Translation (เครื่องแปลภาษา), Question Answering (ระบบถาม-ตอบ) และ Sentence Representation (การแปลงข้อความให้เป็นข้อมูล)
Notebook: [Part1](), [Part2](), [Part3]()


## Track - Tabular Data

### Week 4 - บทที่ 4t

Loren ipsum

### Week 5 - บทที่ 5t

Loren ipsum

### Week 6 - บทที่ 6t

Loren ipsum

## Week 7 - [บทที่ 7 - จริยธรรมปัญญาประดิษฐ์]()

Loren ipsum

## Week 8 - [บทที่ 8 - Prototype Deployment]()

ในบทเรียนนี้จะแนะนำวิธีการ Deploy โปรเจค ML / AI โดยจะนำ source code ที่เขียนไว้ใน notebook มาสร้างเป็นโปรเจค Streamlit, เรียนรู้ widget ต่างๆของ Streamlit เพื่อใช้ทำ Visualization ไม่ว่าจะเป็นการนำผลลัพธ์จากการรันโมเดลมาพล็อตเป็นกราฟ ตาราง แสดงรูปภาพที่เกี่ยวข้อง และอื่นๆ รวมโค้ดทั้งหมดออกเป็นเป็นโปรเจค จากนั้น Deploy โปรเจคไปยัง Heroku, Streamlit Cloud หรือ Cloud Server อื่นๆ เช่น DigitalOcean / AWS / Google Cloud / Azure 

# กิตติกรรมประกาศ - Acknowledgements

ส่วนหนึ่งของบทเรียนของ AI Builders ทำการดัดแปลง-แก้ไข-ต่อเติมจาก [fastai Practical Deep Learning for Coders v4 part1](https://course.fast.ai/) ตามลิขสิทธิ์ GNU General Public License v3.0 เพื่อให้เหมาะแก่นักเรียนผู้ใช้ภาษาไทยเป็นภาษาแรก ได้แก่ บทที่ 1 และ 2 (ปรับแต่งจาก [Lesson 1](https://course.fast.ai/videos/?lesson=1) พร้อมเพิ่มเติมเนื้อหา), 3 (ปรับแต่งจาก [Lesson 3](https://course.fast.ai/videos/?lesson=4) และ [Lesson 4](https://course.fast.ai/videos/?lesson=4)) และ 7 (แปลเป็นภาษาไทยและเพิ่มเติมเนื้อหาจาก [Lesson 5](https://course.fast.ai/videos/?lesson=5))

We adapted and augmented some lessons from [fastai Practical Deep Learning for Coders v4 part1](https://course.fast.ai/) for our curriculum to suit our students whose first language is Thai, namely Lesson 1 and 2 (adapted from [Lesson 1](https://course.fast.ai/videos/?lesson=1); augmented our original contents), Lesson 3 (adapted from [Lesson 3](https://course.fast.ai/videos/?lesson=4) and [Lesson 4](https://course.fast.ai/videos/?lesson=4)), Lesson 7 (translated from [Lesson 5](https://course.fast.ai/videos/?lesson=5) and added localized examples).
