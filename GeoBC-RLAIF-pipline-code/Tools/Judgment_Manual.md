# “Intra-class vs Inter-class Change” Annotation Decision Manual

## 1. Annotation Objective

This manual is used to determine change regions in **bi-temporal remote sensing imagery**.  
The core task is **not simply identifying visual differences**, but determining:

1. Whether the two timestamps belong to the **same Level-2 land-use category**.
2. If the appearance differs visually but the Level-2 land-use category **has not changed**, it should be labeled as **Intra-class Change / Non-target Change**.
3. If the difference is caused by **shadow, illumination, phenology, water accumulation, misregistration, etc.**, it should be labeled as **Pseudo Change**.

---

## 2. Four Required Labels

Each candidate change region must ultimately fall into **one of the following four categories**.

### 1. `NO_CHANGE`

Both timestamps belong to **the same category**, and there is **no evidence of land-use functional change**.

Even if **color, texture, moisture, crop growth, or boundary clarity** changes, it should **not be treated as target change**.

---

### 2. `INTRA_CHANGE`

The two timestamps belong to **different categories within the same higher-level class**.

Examples:

- Within cropland:  
  *paddy field → irrigated cropland → dry farmland*
- Within transportation land:  
  *rural road → other transportation land*

These should be considered **intra-class changes**.

If transitions occur **across major land-use classes**, such as:

- cropland → construction land
- forest → grassland
- water → construction land

then the change is considered **target change**.

---

### 3. `PSEUDO_CHANGE`

Visual differences exist, but **there is no sufficient evidence that land-use categories changed**.

Typical sources:

- shadow variation
- solar angle differences
- phenological changes
- temporary flooding or water recession
- harvesting
- temporary exposure of soil
- sensor differences
- registration errors

---

### 4. `UNCERTAIN`

The class cannot be determined reliably due to:

- ambiguous category boundaries
- insufficient resolution
- severe occlusion
- conflicting evidence

The annotator must:

- explain the reason for uncertainty
- lower the confidence score
- recommend field verification if necessary

---

## 3. General Principle: Classify First, Compare Later

### Principle 1: Classify t1 and t2 independently before judging change

It is **forbidden** to jump directly from *“it looks different”* to *“inter-class change occurred”*.

Correct workflow:

1. classify **t1 independently**
2. classify **t2 independently**
3. determine the change label based on **before vs after class**

---

### Principle 2: Land-use type takes priority over visual appearance

Even if color, moisture, texture, or surface condition changes significantly,  
as long as the **land-use function does not transition**, it should **not be labeled as inter-class change**.

---

### Principle 3: Intermediate states should follow official definitions

For example:

A construction site may appear as **bare land or irregular texture**,  
but if the area is undergoing **construction, mining, or demolition**, it should be classified as:

**Human-disturbed land**

This includes:

- mines
- quarries
- construction sites
- demolition areas
