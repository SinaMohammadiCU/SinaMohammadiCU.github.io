---
title: SQL Learning track
tags: [SQL]
style: fill
color:
description: I am taking notes of my journey in reviewing and learning SQL in this post. I am learning ten commands per day.
date: 06/01/2022
---

Source: I am learning from SQL Cookbook by A. Molinaro and R. Graaf  <br/><br/>
1.1 Retrieving ALL Rows and Columns from a table.
```sql
SELECT * FROM emp
```
1.2 Retrieving a Subset of Rows from a Table
```sql
SELECT * FROM emp
 WHERE deptno = 10
```
Below is the list of common operators:  
= < > <= >= ! <> AND OR

1.3 Finding rows that satisfies multiple conditions.
```sql
SELECT * FROM emp WHERE deptno = 10 OR
comm IS NOT NULL OR sal <= 2000 AND deptno = 20
```
1.4 Retrieving a subset of columns from a table.
```sql
SELECT ename, deptno, sal FROM emp
```

1.5 Providing meaningful names for Columns (or Aliasing).
```sql
SELECT sal AS salary, comm AS commision FROM emp
```

1.6 Referencing an Aliased column in the WHERE clause
```sql
SELECT * FROM (SELECT sal AS salary, commm AS commission FROM emp)
WHERE salary < 5000
```
1.7 Concatenating column values <br/><br/>
use double vertical bar || as Concatenating operator. or instead of || we could use CONCAT.
```sql
SELECT name || "works as a" || job AS msg FROM emp WHERE deptno = 10
```

1.8 Using conditional logic in a SELECT statement
```sql
SELECT ename, sal,
      CASE WHEN sal <= 2000 THEN "UNDERPAID"
            WHEN sal >= 4000 THEN "OVERPAID"
            ELSE "OK"
      END AS STATUS
  FROM emp
```

1.9 Limiting the number of rows returned
```sql
SELECT * FROM emp LIMIT 5
```
1.10 Returning n random records from a table.
```sql
SELECT ename, job FROM emp
ORDER BY RANDOM() LIMIT 5
```
1.11 Finding null values
```sql
SELECT * FROM emp WHERE comm IS NULL
```
Note: You can also use IS NOT NULL to find rwos without a null in a given column 
