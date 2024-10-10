---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

## Education

- Bachelor of Technology, Computer Science
  PES University
  - **Relevant Coursework**: : Data Structures and Algorithms (C++, Java, Python), Object Oriented Programming (Java), DBMS (MySQL, PostgreSQL), Graph Theory (Graph Neural Networks, Neo4j), AR/VR (Blender, GLUT, Unity), Deep Learning (PyTorch, TensorFlow)
  - **Awards**: 3x Distinction
  - **Specialization**: Machine Intelligence and Data Science

## Experience

1. Software Developer at [Nasdaq](https://www.nasdaq.com/)
1. Computer Vision Intern at [StanceBeam](https://www.stancebeam.com/)

## Teaching

  <ul>{% for post in site.teaching reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

## Publications

  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

## Technical Skills

- **Languages**: Python, C++, Java, C
- **Concepts**: Operating System, Artificial Intelligence, Machine Learning, Neural Networks, Database, Agile Methodology, Cloud Computing, Generative AI, Large Language Models, Computer Vision, Data Science, Computer Networks
- **Certifications**: AWS Educate Introduction to Cloud 101, Amazon Web Services; Quantum Computing Using Qiskit, PESU I/O; LFD103, The Linux Foundation

## Articles

  <ul>{% for post in site.talks reversed %}
    {% include archive-single-talk-cv.html  %}
  {% endfor %}</ul>
