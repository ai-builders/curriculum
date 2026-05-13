# curriculum

8-week curriculum for AI Builders

<img src="https://raw.githubusercontent.com/ai-builders/curriculum/refs/heads/main/images/curriculum_2026.png" width="1000">

## สารบัญ

- [บทที่ 0 - How to AI 2026 [Recommended Setup for AI Builders]](#chapter0)
- [บทที่ 1 - Artificial Intelligence (AI) คืออะไร](#chapter1)
- [บทที่ 2 - ชุดข้อมูลมหัศจรรย์และถิ่นที่อยู่](#chapter2)
- [บทที่ 3 - Stochastic Gradient Descent ตั้งแต่เริ่มต้น](#chapter3)
- [Track - Agentic](#track-agentic)
  - [บทที่ 4a - Agentic AI from Scratch](#chapter4a)
  - [บทที่ 5a - RAG and Automating Prompting](#chapter5a)
  - [บทที่ 6a - Programmtic Tool Calling and Other Topics](#chapter6a)
- [Track - Vision](#track-vision)
  - [บทที่ 4v - Image Classification and Semantic Segmentation](#chapter4v)
  - [บทที่ 5v - Object Detection](#chapter5v)
  - [บทที่ 6v - GANs and Advanced Topics](#chapter6v)
- [Track - Texts](#track-texts)
  - [บทที่ 4n - ภาพรวมของ NLP และ Transformer](#chapter4n)
  - [บทที่ 5n - Encoder-only language model และ huggingface](#chapter5n)
  - [บทที่ 6n - Large Language Model and the GPTs](#chapter6n)
- [Track - Tabular Data](#track-tabular)
  - [บทที่ 4t - Introduction to Tabular Data, Correlation and Regression](#chapter4t)
  - [บทที่ 5t - Classification](#chapter5t)
  - [บทที่ 6t - Similarity, Recommendation Clustering](#chapter6t)
- [บทที่ 7 - จริยธรรมปัญญาประดิษฐ์](#chapter7)
- [บทที่ 8 - Prototype Deployment](#chapter8)

<a name="chapter0"></a>

## [บทที่ 0 - How to AI 2026 [Recommended Setup for AI Builders]](https://youtu.be/cquH30vEYGQ)

ในบทเรียนนี้เราจะแนะนำเครื่องมือที่จำเป็นสำหรับการพัฒนา AI ในโครงการ AI Builders 2026 เพื่อให้ทุกคนสามารถทำโครงงานให้เสร็จภายใน 8 สัปดาห์ได้อย่างมีประสิทธิภาพ

Video: [YouTube](https://youtu.be/cquH30vEYGQ)

Notebooks: [TH](notebooks/00_how_to_ai_2026.ipynb)

<a name="chapter1"></a>

## Week 1 - [บทที่ 1 - Artificial Intelligence คืออะไร](https://www.youtube.com/watch?v=cUZdHERY5Rk)

ในบทเรียนนี้เราจะเรียนรู้ว่า Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL) และ Large Language Model (LLM) คืออะไร เหมือนกันหรือแตกต่างกันอย่างไร สามแนวทางการพัฒนาระบบปัญญาประดิษฐ์ที่กำลังเป็นที่นิยมในปัจจุบันคืออะไร-แนวทางไหนเหมาะกับการใช้งานแบบใด เราจะเรียนรู้ส่วนประกอบของระบบปัญญาประดิษฐ์ วิธีการ​ "โปรแกรม" LLM (แทนที่จะแค่ prompt), เทรน machine learning model ด้วยตัวอย่างจำแนกรูปภาพอาหารไทย 48 ชนิดจากชุดข้อมูล [FoodyDudy](https://github.com/GemmyTheGeek/FoodyDudy) หลังจากนั้นเราจะเรียนรู้วิธีการวัดผลเบื้องต้น (ก่อนไปเรียนละเอียดในบทที่ 3) และความท้าทายของการวัดผลในโลกที่ LLM ให้ผลลัพธ์ที่ไม่ตายตัว

Video: [YouTube](https://www.youtube.com/watch?v=cUZdHERY5Rk)

Notebooks: [EN](notebooks/01_ai_what_2026.ipynb), [TH](notebooks/01_ai_what_2026_th.ipynb)

<a name="chapter2"></a>

## Week 2 - [บทที่ 2 - ชุดข้อมูลมหัศจรรย์และถิ่นที่อยู่](https://youtu.be/AQZYox8kZiU)

ในปัจจุบันชุดข้อมูลที่มีพร้อมทั้งปริมาณและคุณภาพเป็นส่วนสำคัญในการสร้าง AI Systems ในบทเรียนนี้เราจะเรียนรู้วิธีการหาข้อมูลมาเทรนโมเดลของเราทั้งจากชุดข้อมูล open data, web scraping, หรือสร้างขึ้นมาเองจากโมเดลและโค้ด open source ทั้งนี้การหาข้อมูลมาเทรนโมเดลจากแหล่งข้อมูลสาธารณะที่กล่าวมานั้นเราต้องให้ความสำคัญเรื่องลิขสิทธิ์และจริยธรรม (แม้แต่โมเดลเองก็สร้างข้อมูลที่ผิดลิขสิทธิ์-จริยธรรมได้; เรียนเพิ่มเติมในบทที่ 7)

Video: [YouTube](https://youtu.be/AQZYox8kZiU)

Notebooks: [EN](notebooks/02_fantastic_datasets_2026.ipynb), [TH](notebooks/02_fantastic_datasets_2026_th.ipynb)

<a name="chapter3"></a>

## Week 3

### [บทที่ 3a - Metrics, Baselines, และการประเมินระบบ LLM](https://youtu.be/4TxqR1h7QZ4)

ในบทเรียนนี้เราจะเจาะลึกเรื่องการวัดผลและการเปรียบเทียบกับ baseline ตั้งแต่การแบ่งข้อมูล train/valid/test split อย่างถูกต้อง, metric สำหรับ classification, regression metrics, information retrieval ไปจนถึงการประเมินระบบ LLM ยุคใหม่ด้วย semantic similarity, LLM-as-Judge และ rubric-based scoring พร้อมกรอบการตัดสินใจเลือก metric ที่เหมาะสมกับแต่ละ task

Video: [YouTube](https://youtu.be/4TxqR1h7QZ4)

Notebooks [EN](notebooks/03a_metrics_and_baselines_2026.ipynb), [TH](notebooks/03a_metrics_and_baselines_2026_th.ipynb)

### [บทที่ 3b - Stochastic Gradient Descent ตั้งแต่เริ่มต้น](https://www.youtube.com/watch?v=1tukMeP_UdE)

ในบทเรียนนี้ เราจะทำการสร้างวิธีที่โมเดลของเราเรียนรู้ในบทเรียนที่แล้วๆมา เรียกว่า stochastic gradient descent ขึ้นมาเองตั้งแต่ต้นโดยใช้เพียงแค่ Pytorch สำหรับ linear algebra และการทำ partial derivatives เท่านั้น ด้วยตัวอย่างการจำแนกรูปภาพตัวเลข 3 และ 7 ออกจากกัน

บทเรียนแปล-สรุปมาจาก [04_mnist_basics.ipynb](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb) ของ [fastai](https://course.fast.ai/) ผู้ที่สนใจสามารถไปติดตามบทเรียนต้นทางได้ที่ [course.fast.ai](https://course.fast.ai/)

Video: [YouTube](https://www.youtube.com/watch?v=1tukMeP_UdE)

Notebooks: [TH](notebooks/03b_sgd_from_scratch.ipynb)

<a name="track-agentic"></a>

## Track - [Agentic](TBA)

<a name="chapter4a"></a>

### Week 4 - 4a [Agentic AI from Scratch](https://youtu.be/9qWrr-1hOyY)

เนื้อหาในบทเรียนนี้:

- ทำความเข้าใจ LLM เบื้องต้น
- รู้จัก ReAct framework สำหรับการสร้าง LLM driven agents
- เทคนิคการใช้ CoT และ Few-shot prompting เพื่อเพิ่มความแม่นยำให้ agents
- ปัญหาที่พบบ่อย (Common failure cases) พร้อมแนวทางแก้ไขในบทเรียน
  โดยเราจะสร้าง agent ในรูปแบบพื้นฐานที่สุดเป็นตัวอย่างด้วย LLM `gemma-3-1b-it`

Video: [YouTube](https://youtu.be/9qWrr-1hOyY)

Notebook: [Agentic AI from Scratch](https://github.com/ai-builders/curriculum/blob/main/notebooks/04a_Building_an_Agent_from_Scratch.ipynb)

### Week 5 - 5a [RAG and Automating Prompting](https://youtu.be/A0bGe_RpbwE)

เนื้อหาในบทเรียนนี้:

- การ deploy LLM ดัวย llama cpp (กับวิธีอื่น ๆ)
- การใช้ LangGraph เขียน Agents
- RAG - ทำให้ agent ค้นข้อมูลใน vector databases
- APE ดัวย DSPy ทำให้ prompt พัฒนา
  โดยเราจะสร้าง agent ที่ดีขี้นเพื่อตอบคำถามที่อยู่นอก training scope

Video: [YouTube](https://youtu.be/A0bGe_RpbwE)

Notebook: [RAG and APE](https://github.com/ai-builders/curriculum/blob/main/notebooks/05a_RAG_and_APE.ipynb)

Disclaimer: due to complex file heirarchy and a need for reproducible environments, please use this [repository](https://github.com/kkuroma/AIB-agentic-notebooks) for local execution.

### Week 6 - 6a [Programmatic Tool Calling and Other Topics](TBA)

เนื้อหาในบทเรียนนี้:

- การเขียน custom tools นอกเหนือจาก RAG 
- รวมส่วนประกอบจาก week ก่อน ๆ (ReAct and RAG) เข้าด้วยกัน

Video: [YouTube TBA](TBA)

Notebook: [RAG and APE](https://github.com/ai-builders/curriculum/blob/main/notebooks/06a_Tool_Calling.ipynb)

Disclaimer: due to complex file heirarchy and a need for reproducible environments, please use this [repository](https://github.com/kkuroma/AIB-agentic-notebooks) for local execution.

<a name="track-vision"></a>

## Track - [Vision](https://www.youtube.com/watch?v=qKP84HVuXyQ&list=PLMkXpBmmcKL-7-Xdq2MwDR1oLtTqaSSxR)

<a name="chapter4v"></a>

### Week 4 - 4v [Image Classification and Semantic Segmentation](https://youtu.be/xIHM2Uq1BXE)

ในบทเรียนนี้เราจะเรียนรู้ว่าการวิเคราะห์ภาพด้วย AI ในหัวข้อ Image Classification (การจำแนกประเภทภาพ) และ Semantic Segmentation (การแบ่งส่วนภาพระดับพิกเซล) มีหลักการทำงานอย่างไร โดยเริ่มตั้งแต่การปูพื้นฐานทฤษฎี Convolutional Neural Network (CNN) และ Vision Transformer ไปจนถึงการลงมือทำจริงด้วยไลบรารี Hugging Face และ PyTorch นอกจากนี้ยังสอนตั้งแต่การเตรียมชุดข้อมูล, การฝึกสอนโมเดล (Train Model), การประเมินผล ไปจนถึงการสร้าง Web Application ง่ายๆ ด้วย Gradio เพื่อทดสอบโมเดลที่เทรนเสร็จแล้ว

Video: [Youtube](https://youtu.be/xIHM2Uq1BXE)

Notebooks: [Image Classification](https://github.com/ai-builders/curriculum/blob/main/notebooks/04v_2026_01_image_classification.ipynb), [Semantic Segmentation](https://github.com/ai-builders/curriculum/blob/main/notebooks/04v_2026_01_semantic_segmentation.ipynb)

<a name="chapter5v"></a>

### Week 5 - 5v [Object Detection]()

TBD

<a name="chapter6v"></a>

### Week 6 - 6v [GANs and Advanced Topics]()

TBD

<a name="track-texts"></a>

## [Track - Texts](https://www.youtube.com/watch?v=x3-57cURLAA&list=PLyyEwPZh6aHrox_MkN9_9UOBHsRQpDBWg&index=1)

Video: [Youtube Playlist](https://www.youtube.com/playlist?list=PLyyEwPZh6aHrox_MkN9_9UOBHsRQpDBWg)
Slide: [pdf](https://github.com/ai-builders/curriculum/releases/download/slides_texts_2025/AI.builder.2025.-.Natural.Language.Processing.pdf)

<a name="chapter4n"></a>

### Week 4 - บทที่ 4n - ภาพรวมของ NLP และ Transformer

ในบทนี้จะแนะนำเห็นภาพรวมของ Natural Language Processing และ Application และเรียนรู้เกี่ยวกับโมเดล Transformer ซึ่งเป็นโมเดล NLP ที่สำคัญที่สุดในขณะนี้

Video:

- [Overview of NLP](https://youtu.be/x3-57cURLAA)
- [Embeddings and word embeddings](https://youtu.be/sYu0UlAcqsc)
- [Transformers model](https://youtu.be/fmzlsM4on5U)

<a name="chapter5n"></a>

### Week 5 - บทที่ 5n - Encoder-only language model และ huggingface

Encoder-only language model เป็นโมเดลที่สามารถปรับจูนเพื่อทำ text classification ที่ดีที่สุด และสะดวกที่สุดตัวหนึ่ง อีกทั้งเป็นพื้นฐานสำหรับโมเดลภาษาขนาดใหญ่ ในบทนี้ผู้เรียนจะได้ทดลองใช้ [Hugging Face](https://huggingface.co/) ([transformers](https://github.com/huggingface/transformers/), [datasets](https://github.com/huggingface/datasets), [tokenizers](https://github.com/huggingface/tokenizers)) เพื่อเทรนโมเดล NLP

Video:

- [Encoder-only model](https://youtu.be/f_mN-lVkZ9I)
- [Huggingface tutorial: Fine-tuning encoder-only models for sequence classification](https://youtu.be/PdKl-wwA6nw)

Notebooks: [Huggingface](https://github.com/ai-builders/curriculum/blob/main/notebooks/05n_NLP_huggingface.ipynb)

<a name="chapter6n"></a>

### Week 6 - บทที่ 6n - Large Language Model and the GPTs

Large language models (LLM) และ ChatGPT เป็นโมเดลที่ทำให้ AI กลับได้รับความนิยมอีกครั้ง และจัดเป็นการปฏิวัติวงการ NLP ครั้งใหญ่ครั้งหนึ่ง บทนี้จะพูดถึงการเทรน Large language model และการเขียน prompt ที่จะทำให้ใช้งาน LLM สำหรับการสร้าง NLP Application

Video:

- [Large language model and the GPT's](https://youtu.be/KMcVK01VO58)
- [Prompt Engineering](https://youtu.be/uxaYbT-crNM)
- [Retrieval Augmented Generation](https://youtu.be/xBmmOkTFs5Y)

Reading: [Chapter 10 Programming for NLP Application](https://attapol.github.io/programming-nlp-book/book/module10/1-large-language-model.html)

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

## Week 7 - [บทที่ 7 - จริยธรรมปัญญาประดิษฐ์](https://youtu.be/Dxs5cZRURi8)

เมื่อปัญญาประดิษฐ์เข้ามามีบทบาทในชีวิตประจำวัน รวมถึงใช้ในการทำงานสาขาต่างๆ อาทิ ช่วยตรวจโรค ช่วยตรวจจับผู้กระทำผิด หรือช่วยตัดสินค่าตอบแทน/บทลงโทษ ฯลฯ จะเห็นได้ว่าปัญญาประดิษฐ์เกี่ยวข้องกับประเด็นทางสังคมและส่งผลกระทบต่อคนเป็นจำนวนมาก บางครั้งปัญญาประดิษฐ์มีการตัดสินใจที่ผิดพลาด ส่งผลกระทบกับชีวิตของคน หลายครั้งปัญญาประดิษฐ์เป็นส่วนหนึ่งของการเผยแพร่อคติโดยที่ผู้พัฒนาคาดไม่ถึง หรือบางกรณีเป็นการจงใจนำปัญญาประดิษฐ์ไปใช้เพื่อการทำร้ายผู้อื่น
การพัฒนาปัญญาประดิษฐ์จึงต้องคำนึงถึงหลักจริยธรรมปัญญาประดิษฐ์หรือ AI Ethics ในการพัฒนาเทคโนโลยีอย่างมีความรับผิดชอบ ในสัปดาห์นี้ เราจะมาทำความเข้าใจว่าปัญญาประดิษฐ์โดยเฉพาะ Generative AI ที่เข้ามามีบทบาทในชีวิตของเราอย่างมากก็สามารถมีอคติในการรับและเผยแพร่ข้อมูลได้อย่างไร รวมถึงคำนึงถึงโอกาสที่เทคโนโลยีจะถูกนำไปใช้ในทางที่ผิดและเราจะหาทางป้องกันความเสี่ยงได้อย่างไร

บทเรียนนี้แปลเป็นภาษาไทยและเพิ่มเติมเนื้อหาจาก [Lesson 5](https://course.fast.ai/videos/?lesson=5) ของ [fastai Practical Deep Learning for Coders v4 part1](https://course.fast.ai/) โดย Rachel Thomas

Video: [YouTube](https://youtu.be/Dxs5cZRURi8)

Slides: [pdf](notebooks/07_ai_ethics_2026.pdf)

<a name="chapter8"></a>

## Week 8 - [บทที่ 8 - Prototype Deployment](https://youtu.be/qa2Sxesdr7Y)

ในบทเรียนนี้จะแนะนำวิธีการ Deploy โปรเจค ML / AI โดยจะนำ source code ที่เขียนไว้ใน notebook มาสร้างเป็นโปรเจค Streamlit, เรียนรู้ widget ต่างๆของ Streamlit เพื่อใช้ทำ Visualization ไม่ว่าจะเป็นการนำผลลัพธ์จากการรันโมเดลมาพล็อตเป็นกราฟ ตาราง แสดงรูปภาพที่เกี่ยวข้อง และอื่นๆ รวมโค้ดทั้งหมดออกเป็นเป็นโปรเจค จากนั้น Deploy โปรเจคไปยัง Heroku, Streamlit Cloud หรือ Cloud Server อื่นๆ เช่น DigitalOcean / AWS / Google Cloud / Azure

Video: [YouTube](https://youtu.be/qa2Sxesdr7Y)

Slides: [Google Slides](https://docs.google.com/presentation/d/1_0oWVo-DPM43QR86evChI9q6FrPVuVfP/), [pdf](https://github.com/ai-builders/curriculum/releases/download/slides_prototype_deployment/08_prototype_deployment.pdf)

# กิตติกรรมประกาศ - Acknowledgements

ส่วนหนึ่งของบทเรียนของ AI Builders ทำการดัดแปลง-แก้ไข-ต่อเติมจาก [fastai Practical Deep Learning for Coders v4 part1](https://course.fast.ai/) ตามลิขสิทธิ์ GNU General Public License v3.0 เพื่อให้เหมาะแก่นักเรียนผู้ใช้ภาษาไทยเป็นภาษาแรก ได้แก่ บทที่ 1 และ 2 (ปรับแต่งจาก [Lesson 1](https://course.fast.ai/videos/?lesson=1) พร้อมเพิ่มเติมเนื้อหา), 3b (ปรับแต่งจาก [Lesson 3](https://course.fast.ai/videos/?lesson=4) และ [Lesson 4](https://course.fast.ai/videos/?lesson=4)) และ 7 (แปลเป็นภาษาไทยและเพิ่มเติมเนื้อหาจาก [Lesson 5](https://course.fast.ai/videos/?lesson=5))

We adapted and augmented some lessons from [fastai Practical Deep Learning for Coders v4 part1](https://course.fast.ai/) for our curriculum to suit our students whose first language is Thai, namely Lesson 1 and 2 (adapted from [Lesson 1](https://course.fast.ai/videos/?lesson=1); augmented our original contents), Lesson 3b (adapted from [Lesson 3](https://course.fast.ai/videos/?lesson=4) and [Lesson 4](https://course.fast.ai/videos/?lesson=4)), Lesson 7 (translated from [Lesson 5](https://course.fast.ai/videos/?lesson=5) and added localized examples).
