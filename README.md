# AtomMem : Learnable Dynamic Agentic Memory with Atomic Memory Operation
> This repo is the official code implementation of **AtomMem : Learnable Dynamic Agentic Memory with Atomic Memory Operation** <br>
> [[arXiv]]() <br>
> **Yupeng Huo, Yaxi Lu, Zhongzhang, Haotian Chen, Yankai Lin**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

⭐️ Please star this repository if you find it helpful!

## Overview
We introduce **AtomMem**, a dynamic memory framework that reframes memory management as a learnable decision-making problem. Instead of relying on predefined pipelines, AtomMem decomposes memory manipulation into **atomic CRUD operations**—Create, Read, Update, and Delete—and trains the agent to decide when and how to invoke these operations based on task context.