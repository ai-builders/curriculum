# curriculum
8-week curriculum for AI Builders

<img src="https://github.com/ai-builders/curriculum/raw/main/images/curriculum.png" width="1000">

## สารบัญ
- [บทที่ 1 - Machine Learning คืออะไร](#chapter1)
- [บทที่ 2 - ชุดข้อมูลมหัศจรรย์และถิ่นที่อยู่](#chapter2)
- [บทที่ 3 - Stochastic Gradient Descent ตั้งแต่เริ่มต้น](#chapter3)
- [Track - Vision](#track-vision)
  - [บทที่ 4v - Image Classification](#chapter4v)
  - [บทที่ 5v - Object Detection](#chapter5v)
  - [บทที่ 6v - GANs and Advanced Topics](#chapter6v)
- [Track - Texts](#track-texts)
  - [บทที่ 4n - NLP คืออะไร? บทเรียนจากอดีตสู่ปัจจุบัน](#chapter4n)
  - [บทที่ 5n - การเทรนโมเดลบน Hugging Face พาร์ท 1](#chapter5n)
  - [บทที่ 6n - การเทรนโมเดลบน Hugging Face พาร์ท 2](#chapter6n)
- [Track - Tabular Data](#track-tabular)
  - [บทที่ 4t - Introduction to Tabular Data, Correlation and Regression](#chapter4t)
  - [บทที่ 5t - Classification](#chapter5t)
  - [บทที่ 6t - Similarity, Recommendation Clustering](#chapter6t)
- [บทที่ 7 - จริยธรรมปัญญาประดิษฐ์](#chapter7)
- [บทที่ 8 - Prototype Deployment](#chapter8)

<a name="chapter1"></a>
## Week 1 - [บทที่ 1 - Machine Learning คืออะไร](https://youtu.be/4-PHm7-XIjE)

ในบทเรียนนี้เราจะเรียนรู้ว่า Artificial Intelligence (AI), Machine Learning (ML) และ Deep Learning (DL) คืออะไร เหมือนกันหรือแตกต่างกันอย่างไร เราจะเรียนรู้ส่วนประกอบของระบบ machine learning และวิธีการเทรน machine learning model ด้วยตัวอย่างจำแนกรูปภาพอาหารไทย 48 ชนิดจากชุดข้อมูล [FoodyDudy](https://github.com/GemmyTheGeek/FoodyDudy) หลังจากนั้นเราจะเห็นว่าส่วนประกอบและวิธีการเทรนนี้ถูกใช้กับข้อมูลชนิดอื่นๆ เช่น ข้อความ (texts) และตาราง (tabular data) ได้อย่างไรบ้าง

บทเรียนนี้ปรับแต่งและเพิ่มเติมจาก [fastai Practical Deep Learning for Coders v4 part1 - Lesson 1](https://course.fast.ai/videos/?lesson=1) เพื่อให้เหมาะกับโครงการ AI Builders

Video: [YouTube](https://youtu.be/4-PHm7-XIjE)

Notebooks: [All Parts](notebooks/01_ml_what.ipynb)

<a name="chapter2"></a>
## Week 2 - [บทที่ 2 - ชุดข้อมูลมหัศจรรย์และถิ่นที่อยู่](https://youtu.be/1djko0F3auA)

ในปัจจุบันชุดข้อมูลที่มีพร้อมทั้งปริมาณและคุณภาพเป็นส่วนสำคัญในการสร้าง ML models ในบทเรียนนี้เราจะเรียนรู้วิธีการหาข้อมูลมาเทรนโมเดลของเราทั้งจากชุดข้อมูล open data, web scraping, หรือสร้างขึ้นมาเองจากโมเดลและโค้ด open source ทั้งนี้การหาข้อมูลมาเทรนโมเดลจากแหล่งข้อมูลสาธารณะที่กล่าวมานั้นเราต้องให้ความสำคัญเรื่องลิขสิทธิ์และจริยธรรม (แม้แต่โมเดลเองก็สร้างข้อมูลที่ผิดลิขสิทธิ์-จริยธรรมได้; เรียนเพิ่มเติมในบทที่ 7)

Video: [YouTube](https://youtu.be/1djko0F3auA)

Notebooks: [All Parts](notebooks/02_fantastic_datasets.ipynb)

<a name="chapter3"></a>
## Week 3 - [บทที่ 3 - Stochastic Gradient Descent ตั้งแต่เริ่มต้น](https://www.youtube.com/watch?v=1tukMeP_UdE) 

ในบทเรียนนี้ เราจะทำการสร้างวิธีที่โมเดลของเราเรียนรู้ในบทเรียนที่แล้วๆมา เรียกว่า stochastic gradient descent ขึ้นมาเองตั้งแต่ต้นโดยใช้เพียงแค่ Pytorch สำหรับ linear algebra และการทำ partial derivatives เท่านั้น ด้วยตัวอย่างการจำแนกรูปภาพตัวเลข 3 และ 7 ออกจากกัน

บทเรียนแปล-สรุปมาจาก [04_mnist_basics.ipynb](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb) ของ [fastai](https://course.fast.ai/) ผู้ที่สนใจสามารถไปติดตามบทเรียนต้นทางได้ที่ [course.fast.ai](https://course.fast.ai/)

Video: [YouTube](https://www.youtube.com/watch?v=1tukMeP_UdE)

Notebooks: [All Parts](notebooks/03_sgd_from_scratch.ipynb)

<a name="track-vision"></a>
## Track - [Vision](https://www.youtube.com/watch?v=qKP84HVuXyQ&list=PLMkXpBmmcKL-7-Xdq2MwDR1oLtTqaSSxR) 

<a name="chapter4v"></a>
### Week 4 - 4v [Image Classification](https://www.youtube.com/watch?v=qKP84HVuXyQ) 

ในบทเรียนนี้เราจะมาลองสร้างโมเดล Image classification เพื่อแยกพันธุ์ของน้องหมาโดยใช้เทคนิค Transfer learning ด้วยไลบรารี่ต่างๆ ได้แก่ FastAI, Pytorch และ Pytorch Lightning นอกจากนั้นเราจะมาดูองค์ประกอบของการใช้ Pytorch และการใช้ Image augmentation ด้วยไลบรารี่ `torchvision`

Video: [Part 1](https://www.youtube.com/watch?v=qKP84HVuXyQ), [Part 2](https://www.youtube.com/watch?v=ZRLT4G48EZQ), [Part 3](https://www.youtube.com/watch?v=h7hhfmFJj2c), [Part 4](https://www.youtube.com/watch?v=jYNG9hWvcB0)

Slides (หน้า 1-33): [Google slide](https://docs.google.com/presentation/d/1uSyldRREnvHu1zdI2Etczk2BQaOAoG1Z7UFrPkg-xZs/edit?usp=sharing), [pdf](https://github.com/ai-builders/curriculum/releases/download/slides_vision/04v-06v_slides.pdf)

Notebooks: [Part 1](https://github.com/ai-builders/curriculum/blob/main/notebooks/04v_classification.ipynb), [Part 2](https://github.com/ai-builders/curriculum/blob/main/notebooks/04v_classification_pytorch.ipynb) 

<a name="chapter5v"></a>
### Week 5 - 5v [Object Detection]() 

ในบทเรียนนี้เราจะลองสร้างโมเดล Object detection ด้วยเทคนิค Transfer learning โดยใช้ไลบรารี่ FastAI และ Pytorch กัน เราจะมาดูหน้าตาของการสร้างชุดข้อมูล Object detection และไปดูเครื่องมือต่างๆที่ใช้สร้างชุดข้อมูล Object detection

Video: [Part 1](https://www.youtube.com/watch?v=kCIdKpOZy9M), [Part 2](https://www.youtube.com/watch?v=O_nOvWinsOs), [Part 3](https://youtu.be/Jjr0WmcoyQs)

Slides (หน้า 34-44): [Google slide](https://docs.google.com/presentation/d/1uSyldRREnvHu1zdI2Etczk2BQaOAoG1Z7UFrPkg-xZs/edit?usp=sharing), [pdf](https://github.com/ai-builders/curriculum/releases/download/slides_vision/04v-06v_slides.pdf)

Notebooks: [Object Detection](https://github.com/ai-builders/curriculum/blob/main/notebooks/05v_object_detection.ipynb), [Semantic Segmentation](https://github.com/ai-builders/curriculum/blob/main/notebooks/05v_semantic_segmentation.ipynb)

<a name="chapter6v"></a>
### Week 6 - 6v [GANs and Advanced Topics]() 

ในบทเรียนนี้เราจะมาลองสร้างแบบจำลองประเภท Generative Adversarial Network หรือ GAN แบบเบื้องต้นเพื่อที่จะสร้างภาพเสมือนขึ้นมา (จากข้อมูลภาพหน้าของบุคคลที่มีชื่อเสียง) ด้วยไลบรารี่ต่างๆทั้งจาก PyTorch และ FastAI นอกจากนั้นเรายังจะได้ทำความรู้จักกับ DCGAN ซึ่งเป็น GAN แบบเฉพาะทางทางหนึ่งอีกด้วย

Video: [Part 1](https://www.youtube.com/watch?v=Jo6mj-b7Y2M), [Part 2](https://youtu.be/gfAgHWCvhIE), [Part 3](https://youtu.be/Mj8Re4aHxSI)

Slides (หน้า 45-48): [Google slide](https://docs.google.com/presentation/d/1uSyldRREnvHu1zdI2Etczk2BQaOAoG1Z7UFrPkg-xZs/edit?usp=sharing), [pdf](https://github.com/ai-builders/curriculum/releases/download/slides_vision/04v-06v_slides.pdf)

Notebooks: [GAN fastai](notebooks/06v_GAN_fastai.ipynb), [DCGAN_PyTorch](notebooks/06v_DCGAN_pytorch.ipynb)

<a name="track-texts"></a>
## [Track - Texts](https://www.youtube.com/watch?v=EojcWa7mgiM&list=PLMkXpBmmcKL_HxPVAz4-wT-kCiBR4iOSK) 

<a name="chapter4n"></a>
### Week 4 - [บทที่ 4n - NLP คืออะไร? บทเรียนจากอดีตสู่ปัจจุบัน](https://www.youtube.com/watch?v=EojcWa7mgiM) 

ในบทนี้เราจะเรียนเกี่ยวกับ NLP ตั้งแต่พื้นฐาน ไปจนถึง NLP ในยุคปัจจุบันว่ามีการพัฒนาไปอย่างไรบ้าง พร้อมทั้งยกตัวอย่างการทำ text classification (การจำแนกข้อความ) ด้วยวิธีตั้งแต่อดีตยันปัจจุบัน

Video: [YouTube](https://www.youtube.com/watch?v=EojcWa7mgiM)

Slides: [Google Slides](https://docs.google.com/presentation/d/1jXdjU0LqgIzpbfqy8pqI8-zJUc16_0EY8iiAgxcHMyc/edit?usp=sharing), [pdf](https://github.com/ai-builders/curriculum/releases/download/slides_texts/04n_nlp_intro.pdf)

Notebooks: [All Parts](notebooks/04n_NLP_Thai_Text_Classification.ipynb)

<a name="chapter5n"></a>
### Week 5 - [บทที่ 5n - การเทรนโมเดลบน Hugging Face พาร์ท 1](https://youtu.be/Pd9TFBhmx8I) 

ในปัจจุบันการทำงานด้าน NLP มักจะนิยมใช้งาน Deep Learning ในการแก้ปัญหาโจทย์ที่มีความซับซ้อนสูง โดย Library ที่เป็นที่นิยมในปัจจุบันคือ [Hugging Face](https://huggingface.co/) ([transformers](https://github.com/huggingface/transformers/), [datasets](https://github.com/huggingface/datasets), [tokenizers](https://github.com/huggingface/tokenizers)) โดยในบทเรียนนี้เราจะมาเรียนการใช้งาน Hugging Face เพื่อเทรนโมเดลในงานด้าน NLP!

Video: [YouTube](https://youtu.be/Pd9TFBhmx8I)

Slides: [Google Slides](https://docs.google.com/presentation/d/1KFzS5iL4pi-XV_QxtP9G0sxSSy0PqXvuKsDLSPDhZtA/edit?usp=sharing), [pdf](https://github.com/ai-builders/curriculum/releases/download/slides_texts/05n_nlp_finetuning_part1.pdf)

Notebooks: [Part 1](notebooks/05n_NLP_on_HF_01.ipynb), [Part 2](notebooks/05n_NLP_on_HF_02.ipynb), [Part 3](notebooks/05n_NLP_on_HF_03.ipynb)

<a name="chapter6n"></a>
### Week 6 - [บทที่ 6n - การเทรนโมเดลบน Hugging Face พาร์ท 2](https://youtu.be/qB0Mq1uLFZA) 

หลังจากที่เราได้เรียนรู้พื้นฐานของการใช้งาน [Hugging Face](https://huggingface.co/) แล้ว เราจะมาเทรนโมเดลเพื่องานที่ซับซ้อนมากขึ้น เช่น Machine Translation (เครื่องแปลภาษา), Question Answering (ระบบถาม-ตอบ) และ Sentence Representation (การแปลงข้อความให้เป็นข้อมูล)

Video: [YouTube](https://youtu.be/qB0Mq1uLFZA)

Slides: [Google Slides](https://docs.google.com/presentation/d/1lpnF9ZteWukd5RVBmf1mN9ZKaUoxhNByD0rTAeEqp00/edit?usp=sharing), [pdf](https://github.com/ai-builders/curriculum/releases/download/slides_texts/06n_nlp_finetuning_part2.pdf)

Notebooks: [Part 1](notebooks/06n_Machine_Translation.ipynb), [Part 2](notebooks/06n_Question_Answering.ipynb), [Part 3](notebooks/06n_Sentence_Representation.ipynb)

<a name="track-tabular"></a>
## [Track - Tabular Data](https://www.youtube.com/watch?v=M0Gv5P_n_gk&list=PLMkXpBmmcKL9ecmWbKKOsbXcOSilqpUJ_) 

<a name="chapter4t"></a>
### Week 4 - [บทที่ 4t - Introduction to Tabular Data, Correlation and Regression](https://www.youtube.com/watch?v=X0B0HoK-b1Q) 

การสร้างสมการความสัมพันธ์ (correlation) เพื่อทำนายตัวแปรประเภทตัวเลข (numerical) เพื่อนำไปใช้ในการหาความสัมพันธ์หรือพยากรณ์ เช่น การหาความสัมพันธ์ระหว่างตัวแปรที่มีผลต่อยอดขาย หรือ ทำนายพยากรณ์ยอดขายในอนาคต

Video: [Part 1](https://www.youtube.com/watch?v=X0B0HoK-b1Q), [Part 2](https://www.youtube.com/watch?v=m0r4W0HlqVM), [Part 3](https://www.youtube.com/watch?v=M0Gv5P_n_gk)

Notebooks: [Part 1](https://github.com/ai-builders/curriculum/blob/main/notebooks/04t_01_intro_to_tabular_data.ipynb), [Part 2](https://github.com/ai-builders/curriculum/blob/main/notebooks/04t_02_correlation_regression.ipynb), [Part 3](https://github.com/ai-builders/curriculum/blob/main/notebooks/04t_03_time_series.ipynb)

<a name="chapter5t"></a>
### Week 5 - [บทที่ 5t - Classification](https://www.youtube.com/watch?v=Dz_faAOHQPE) 

การสร้างสมการความสัมพันธ์ เพื่อทำนายตัวแปรประเภทกลุ่ม/ชนิด (categorical) เพื่อใช้ในการทำนายหรือเลือกทางเลือก เช่น ทำนายว่าลูกค้าคนไหนจะหยุดใช้บริการ ทำนายว่าลูกค้าคนไหน เมื่อส่งคูปองไปแล้วจะใช้ หรือ ทำนายว่าเครื่องจักรจะเสียหรือไม่

Video: [Part 1](https://www.youtube.com/watch?v=Dz_faAOHQPE), [Part 2](https://www.youtube.com/watch?v=UFl9itjyzcU)

Notebooks: [Part 1](https://github.com/ai-builders/curriculum/blob/main/notebooks/05t_01_classification.ipynb), [Part 2](https://github.com/ai-builders/curriculum/blob/main/notebooks/05t_02_classification_evaluation.ipynb)

<a name="chapter6t"></a>
### Week 6 - [บทที่ 6t - Similarity, Recommendation and Clustering](https://youtu.be/Qnmynr6gVnw) 

การวิเคราะห์ความคลายคลึงและการแบ่งกลุ่มข้อมูล เพื่อนำไปใช้ในการแนะนำสินค้าหรือเนื้อหาที่ลูกค้าสนใจ เช่น Shopee แนะนำสินค้าที่เราสนใจ หรือ spotify แนะนำเพลงที่ผู้ฟังน่าจะอยากฟังต่อไป รวมถึงการนำข้อมูลมาใช้ในการแบ่งกลุ่มลูกค้าที่มีความสนใจเหมือนกันสำหรับนำไปทำการตลาดเฉพาะกลุ่ม

Video: [Part 1](https://youtu.be/Qnmynr6gVnw), [Part 2](https://www.youtube.com/watch?v=EQvPvSCpOyE)

Notebooks: [Part 1](notebooks/06t_01_similarity_matching.ipynb), [Part 2](notebooks/06t_02_clustering.ipynb)

<a name="chapter7"></a>
## Week 7 - [บทที่ 7 - จริยธรรมปัญญาประดิษฐ์](https://github.com/ai-builders/curriculum/releases/download/slides_ai_ethics/07_ai_ethics.pdf) 

เมื่อปัญญาประดิษฐ์เข้ามามีบทบาทในชีวิตประจำวัน รวมถึงใช้ในการทำงานสาขาต่างๆ อาทิ ช่วยตรวจโรค ช่วยตรวจจับผู้กระทำผิด หรือช่วยตัดสินค่าตอบแทน/บทลงโทษ ฯลฯ จะเห็นได้ว่าปัญญาประดิษฐ์เกี่ยวข้องกับประเด็นทางสังคมและส่งผลกระทบต่อคนเป็นจำนวนมาก บางครั้งปัญญาประดิษฐ์มีการตัดสินใจที่ผิดพลาด ส่งผลกระทบกับชีวิตของคน หลายครั้งปัญญาประดิษฐ์เป็นส่วนหนึ่งของการเผยแพร่อคติโดยที่ผู้พัฒนาคาดไม่ถึง หรือบางกรณีเป็นการจงใจนำปัญญาประดิษฐ์ไปใช้เพื่อการทำร้ายผู้อื่น 
การพัฒนาปัญญาประดิษฐ์จึงต้องคำนึงถึงหลักจริยธรรมปัญญาประดิษฐ์หรือ AI Ethics ในการพัฒนาเทคโนโลยีอย่างมีความรับผิดชอบ ในสัปดาห์นี้ เราจะมาทำความเข้าใจว่าปัญญาประดิษฐ์ก็สามารถมีอคติในการรับและเผยแพร่ข้อมูลได้อย่างไร รวมถึงคำนึงถึงโอกาสที่เทคโนโลยีจะถูกนำไปใช้ในทางที่ผิดและเราจะหาทางป้องกันความเสี่ยงได้อย่างไร

บทเรียนนี้แปลเป็นภาษาไทยและเพิ่มเติมเนื้อหาจาก [Lesson 5](https://course.fast.ai/videos/?lesson=5) ของ [fastai Practical Deep Learning for Coders v4 part1](https://course.fast.ai/) โดย Rachel Thomas

Video: [YouTube](https://www.youtube.com/watch?v=fJoKlpIl8E8)

Slides: [pdf](https://github.com/ai-builders/curriculum/releases/download/slides_ai_ethics/07_ai_ethics.pdf)

<a name="chapter8"></a>
## Week 8 - [บทที่ 8 - Prototype Deployment]() 

ในบทเรียนนี้จะแนะนำวิธีการ Deploy โปรเจค ML / AI โดยจะนำ source code ที่เขียนไว้ใน notebook มาสร้างเป็นโปรเจค Streamlit, เรียนรู้ widget ต่างๆของ Streamlit เพื่อใช้ทำ Visualization ไม่ว่าจะเป็นการนำผลลัพธ์จากการรันโมเดลมาพล็อตเป็นกราฟ ตาราง แสดงรูปภาพที่เกี่ยวข้อง และอื่นๆ รวมโค้ดทั้งหมดออกเป็นเป็นโปรเจค จากนั้น Deploy โปรเจคไปยัง Heroku, Streamlit Cloud หรือ Cloud Server อื่นๆ เช่น DigitalOcean / AWS / Google Cloud / Azure 

# กิตติกรรมประกาศ - Acknowledgements

ส่วนหนึ่งของบทเรียนของ AI Builders ทำการดัดแปลง-แก้ไข-ต่อเติมจาก [fastai Practical Deep Learning for Coders v4 part1](https://course.fast.ai/) ตามลิขสิทธิ์ GNU General Public License v3.0 เพื่อให้เหมาะแก่นักเรียนผู้ใช้ภาษาไทยเป็นภาษาแรก ได้แก่ บทที่ 1 และ 2 (ปรับแต่งจาก [Lesson 1](https://course.fast.ai/videos/?lesson=1) พร้อมเพิ่มเติมเนื้อหา), 3 (ปรับแต่งจาก [Lesson 3](https://course.fast.ai/videos/?lesson=4) และ [Lesson 4](https://course.fast.ai/videos/?lesson=4)) และ 7 (แปลเป็นภาษาไทยและเพิ่มเติมเนื้อหาจาก [Lesson 5](https://course.fast.ai/videos/?lesson=5))

We adapted and augmented some lessons from [fastai Practical Deep Learning for Coders v4 part1](https://course.fast.ai/) for our curriculum to suit our students whose first language is Thai, namely Lesson 1 and 2 (adapted from [Lesson 1](https://course.fast.ai/videos/?lesson=1); augmented our original contents), Lesson 3 (adapted from [Lesson 3](https://course.fast.ai/videos/?lesson=4) and [Lesson 4](https://course.fast.ai/videos/?lesson=4)), Lesson 7 (translated from [Lesson 5](https://course.fast.ai/videos/?lesson=5) and added localized examples).
