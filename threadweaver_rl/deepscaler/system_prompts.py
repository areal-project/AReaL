"""System prompts for DeepScaler repo."""

DEEPSEEK_MATH_SYSTEM_PROMPT = """Let's think step by step and output the final answer within \\boxed{}. """

# For Math ORM to verify correctness of LLM's solution. We disable this by default, as it doesn't help much.
ORM_PROMPT = """You are an expert in verifying if two math answers are the same.
Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are mathematically equivalent.
Your task is to determine if two mathematical answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical mathematical values or expressions, even when written in different forms or notations.

Guidelines for equivalence:
- Different forms of the same number (e.g., 0.5 = 1/2 = 50%)
- Algebraically equivalent expressions (e.g., (x+1)^2 = x^2 + 2x + 1)
- Geometrically equivalent expressions (e.g., r²π = πr²)
- Trigonometrically equivalent expressions (e.g., sin²θ + cos²θ = 1)
- Semantic equivalence (e.g., "impossible" and "no possible solution")
- Different formats of the same solution (e.g., (1,1,1,3) and a=1,b=1,c=1,p=3)
- Solutions with different or no units (e.g., 100 versus 100 degrees)
- For other cases, please use your best judgement to determine if two answers are truly equivalent.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]

-----
Examples:
Problem: What is the area of a circle with radius 2?
Answer 1: 4π
Answer 2: πr² where r=2
Explanation: Answer 2 simplifies to 4π, making both answers identical.
[[YES]]

Problem: Solve for x: x² + 2x + 1 = 0
Answer 1: x = -1
Answer 2: x = -1 ± 0
Explanation: While Answer 2 includes ± 0, this reduces to just -1, making them equivalent.
[[YES]]

Problem: Find all positive integers $a,b,c$ and prime $p$ satisfying that\n\\[ 2^a p^b=(p+2)^c+1.\\]
Answer 1: a=1, b=1, c=1, p=3
Answer 2:  (1, 1, 1, 3)
Explanation: Both answers represent exactly the same solution, just written in different formats. Answer 1 writes out the values with variable names (a=1, b=1, c=1, p=3) while Answer 3 presents them as an ordered tuple (1, 1, 1, 3).
[[YES]]

Problem: The sides of a $99$ -gon are initially colored so that consecutive sides are red, blue, red, blue,..., red, blue, yellow. We make a sequence of modifications in the coloring, changing the color of one side at a time to one of the three given colors (red, blue, yellow), under the constraint that no two adjacent sides may be the same color. By making a sequence of such modifications, is it possible to arrive at the coloring in which consecutive sides \nare red, blue, red, blue, red, blue,..., red, yellow, blue?
Answer 1: There is no such coloring.
Answer 2: It is impossible to perform a series of such modifications that change the start sequence to the end sequence.
Explanation: Both answers are equivalent because they both state that it is impossible to perform a series of such modifications.
[[YES]]

Problem: Find the slope of the line y = 2x + 1
Answer 1: 2
Answer 2: 3
Explanation: These are different numbers and cannot be equivalent.
[[NO]]
-----
"""

# Judge difficulty of the math problem. For labelling difficulty of math problem between 1-10.
MATH_DIFFICULTY_PROMPT = """You will be given a math problem. Your job is to grade the difficulty level from 1-10 according to the AoPS standard.
  Here is the standard:

  All levels are estimated and refer to averages. The following is a rough standard based on the USA tier system AMC 8 - AMC 10 - AMC 12 - AIME - USAMO/USAJMO - IMO, 
  representing Middle School - Junior High - High School - Challenging High School - Olympiad levels. Other contests can be interpolated against this. 
  Notes: 
  Multiple choice tests like AMC are rated as though they are free-response. Test-takers can use the answer choices as hints, and so correctly answer more AMC questions than Mathcounts or AIME problems of similar difficulty. 
  Some Olympiads are taken in 2 sessions, with 2 similarly difficult sets of questions, numbered as one set. For these the first half of the test (questions 1-3) is similar difficulty to the second half (questions 4-6). 
  Scale 
  1: Problems strictly for beginner, on the easiest elementary school or middle school levels (MOEMS, MATHCOUNTS Chapter, AMC 8 1-20, AMC 10 1-10, AMC 12 1-5, and others that involve standard techniques introduced up to the middle school level), most traditional middle/high school word problems. 
  2: For motivated beginners, harder questions from the previous categories (AMC 8 21-25, harder MATHCOUNTS States questions, AMC 10 11-20, AMC 12 5-15, AIME 1-3), traditional middle/high school word problems with extremely complex problem solving. 
  3: Advanced Beginner problems that require more creative thinking (harder MATHCOUNTS National questions, AMC 10 21-25, AMC 12 15-20, AIME 4-6). 
  4: Intermediate-level problems (AMC 12 21-25, AIME 7-9). 
  5: More difficult AIME problems (10-12), simple proof-based Olympiad-style problems (early JBMO questions, easiest USAJMO 1/4). 
  6: High-leveled AIME-styled questions (13-15). Introductory-leveled Olympiad-level questions (harder USAJMO 1/4 and easier USAJMO 2/5, easier USAMO and IMO 1/4). 
  7: Tougher Olympiad-level questions, may require more technical knowledge (harder USAJMO 2/5 and most USAJMO 3/6, extremely hard USAMO and IMO 1/4, easy-medium USAMO and IMO 2/5). 
  8: High-level Olympiad-level questions (medium-hard USAMO and IMO 2/5, easiest USAMO and IMO 3/6). 
  9: Expert Olympiad-level questions (average USAMO and IMO 3/6). 
  10: Historically hard problems, generally unsuitable for very hard competitions (such as the IMO) due to being exceedingly tedious, long, and difficult (e.g. very few students are capable of solving on a worldwide basis). 
  Examples 
  For reference, here are problems from each of the difficulty levels 1-10: 
  1: Jamie counted the number of edges of a cube, Jimmy counted the numbers of corners, and Judy counted the number of faces. They then added the three numbers. What was the resulting sum? 
  1: Let trapezoid $ABCD$ be such that $AB||CD$. Additionally, $AC = AD = 5$, $CD = 6$, and $AB = 3$. Find $BC$. 
  1: How many integer values of $x$ satisfy $|x| < 3\\pi$? 
  1: The whole number $N$ is divisible by $7$. $N$ leaves a remainder of $1$ when divided by $2,3,4,$ or $5$. What is the smallest value that $N$ can be? 
  1: The value of a two-digit number is $10$ times more than the sum of its digits. The units digit is 1 more than twice the tens digit. Find the two-digit number. 
  1: The coordinates of $\\triangle ABC$ are $A(5,7)$, $B(11,7)$, and $C(3,y)$, with $y>7$. The area of $\\triangle ABC$ is 12. What is the value of $y$? 
  1: How many different 3-digit whole numbers can be formed using the digits 4, 7, and 9, assuming that no digit can be repeated in a number? 
  1.5: A number is called flippy if its digits alternate between two distinct digits. For example, $2020$ and $37373$ are flippy, but $3883$ and $123123$ are not. How many five-digit flippy numbers are divisible by $15?$
  1.5: A rectangular box has integer side lengths in the ratio $1: 3: 4$. Which of the following could be the volume of the box? 
  1.5: Two lines with slopes $\\tfrac14$ and $\\tfrac54$ intersect at $(1,1)$. What is the area of the triangle formed by these two lines and the vertical line $x = 5$? 
  1.5: What is the value of \\[\\log_{3}^{7}\\cdot\\log_{5}^{9}\\cdot\\log_{7}^{11}\\cdot\\log_{9}^{13}\\cdots\\log_{21}^{25}\\cdot\\log_{23}^{27}?\\]
  2: A fair $6$-sided die is repeatedly rolled until an odd number appears. What is the probability that every even number appears at least once before the first occurrence of an odd number? 
  2: A small airplane has $4$ rows of seats with $3$ seats in each row. Eight passengers have boarded the plane and are distributed randomly among the seats. A married couple is next to board. What is the probability there will be $2$ adjacent seats in the same row for the couple? 
  2: Suppose that $\\tfrac{2009}{2014} + \\tfrac{2019}{n} = \\tfrac{a}{b}$, where $a$, $b$, and $n$ are positive integers with $\\tfrac{a}{b}$ in lowest terms. What is the sum of the digits of the smallest positive integer $n$ for which $a$ is a multiple of 1004? 
  2.5: $A$, $B$, $C$ are three piles of rocks. The mean weight of the rocks in $A$ is $40$ pounds, the mean weight of the rocks in $B$ is $50$ pounds, the mean weight of the rocks in the combined piles $A$ and $B$ is $43$ pounds, and the mean weight of the rocks in the combined piles $A$ and $C$ is $44$ pounds. What is the greatest possible integer value for the mean in pounds of the rocks in the combined piles $B$ and $C$?
  2.5: For some positive integer $k$, the repeating base-$k$ representation of the (base-ten) fraction $\\frac{7}{51}$ is $0.\\overline{23}_k = 0.232323..._k$. What is $k$? 
  3: Triangle $ABC$ with $AB=50$ and $AC=10$ has area $120$. Let $D$ be the midpoint of $\\overline{AB}$, and let $E$ be the midpoint of $\\overline{AC}$. The angle bisector of $\\angle BAC$ intersects $\\overline{DE}$ and $\\overline{BC}$ at $F$ and $G$, respectively. What is the area of quadrilateral $FDBG$? 
  3: Wayne has 3 green buckets, 3 red buckets, 3 blue buckets, and 3 yellow buckets. He randomly distributes 4 hockey pucks among the green buckets, with each puck equally likely to be put in each bucket. Similarly, he distributes 3 pucks among the red buckets, 2 pucks among the blue buckets, and 1 puck among the yellow buckets. Once he is ﬁnished, what is the probability that a green bucket contains more pucks than each of the other 11 buckets? 
  3: An object in the plane moves from one lattice point to another. At each step, the object may move one unit to the right, one unit to the left, one unit up, or one unit down. If the object starts at the origin and takes a ten-step path, how many different points could be the final point? 
  3: Consider the integer\\[N = 9 + 99 + 999 + 9999 + \\cdots + \\underbrace{99\\ldots 99}_\\text{321 digits}.\\]Find the sum of the digits of $N$. 
  3: Let $\\triangle LMN$ have side lengths $LM = 15$, $MN = 14$, and $NL = 13$. Let the angle bisector of $\\angle MLN$ meet the circumcircle of $\\triangle LMN$ at a point $T \\ne L$. Determine the area of $\\triangle LMT$.  
  3.5: Find all three-digit numbers $abc$ (with $a \\neq 0$) such that $a^{2} + b^{2} + c^{2}$ is a divisor of 26. 
  3.5: Consider polynomials $P(x)$ of degree at most $3$, each of whose coefficients is an element of $\\{0, 1, 2, 3, 4, 5, 6, 7, 8, 9\\}$. How many such polynomials satisfy $P(-1) = -9$?   
  3.5: Find the number of integer values of $k$ in the closed interval $[-500,500]$ for which the equation $\log(kx)=2\log(x+2)$ has exactly one real solution.
  3.5: In a drawer, there are at most $2009$ balls, some of them are white, the rest are blue, which are randomly distributed. If two balls were taken at the same time, then the probability that the balls are both blue or both white is $\\frac12$. Determine the maximum amount of white balls in the drawer, such that the probability statement is true?   
  3.5: Find three isosceles triangles, no two of which are congruent, with integer sides, such that each triangle’s area is numerically equal to 6 times its perimeter.
  4: Define a sequence recursively by $x_0=5$ and\\[x_{n+1}=\\frac{x_n^2+5x_n+4}{x_n+6}\\]for all nonnegative integers $n.$ Let $m$ be the least positive integer such that\\[x_m\\leq 4+\\frac{1}{2^{20}}.\\]In which of the following intervals does $m$ lie? 
  4: An ant makes a sequence of moves on a cube where a move consists of walking from one vertex to an adjacent vertex along an edge of the cube. Initially the ant is at a vertex of the bottom face of the cube and chooses one of the three adjacent vertices to move to as its first move. For all moves after the first move, the ant does not return to its previous vertex, but chooses to move to one of the other two adjacent vertices. All choices are selected at random so that each of the possible moves is equally likely. The probability that after exactly $8$ moves that ant is at a vertex of the top face on the cube is $\\frac{m}{n}$, where $m$ and $n$ are relatively prime positive integers. Find $m + n.$  
  4: Find all real numbers $a,b,c,d$ such that \\[\\left\\{\\begin{array}{cc}a+b+c+d = 20,\\ ab+ac+ad+bc+bd+cd = 150.\\end{array}\\right.\\] 
  $\\textbf{(A) } [9,26] \\qquad\\textbf{(B) } [27,80] \\qquad\textbf{(C) } [81,242]\\qquad\\textbf{(D) } [243,728] \\qquad\\textbf{(E) } [729,\\infty)$ 
  4: The vertices of an equilateral triangle lie on the hyperbola $xy=1$, and a vertex of this hyperbola is the centroid of the triangle. What is the square of the area of the triangle?
  4. Isosceles trapezoid ABCD has parallel sides $AD$ and $BC$, with $BC<AD$ and $AB=CD$. There is a point P on the plane such that $PA=1$,$PB=2$,$PC=3$, and $PD=4$. What is $BC/AD$? 
  4.5: Find, with proof, all positive integers $n$ for which $2^n + 12^n + 2011^n$ is a perfect square.
  4.5: Find the lowest possible values from the function\\[f(x) = x^{2008} - 2x^{2007} + 3x^{2006} - 4x^{2005} + 5x^{2004} - \\cdots - 2006x^3 + 2007x^2 - 2008x + 2009\\]for any real numbers $x$.   
  4.5: Show that the equation $a^{2}b^{2} + b^{2}c^{2} + 3b^{2} - c^{2} - a^{2} = 2005$ has no integer solutions.  
  5: Triangle $ABC$ has side lengths $AB=7,BC=8,$ and $CA=9.$ Circle $\\omega_1$ passes through $B$ and is tangent to line $AC$ at $A.$ Circle $\\omega_2$ passes through $C$ and is tangent to line $AB$ at $A.$ Let $K$ be the intersection of circles $\\omega_1$ and $\\omega_2$ not equal to $A.$ Then $AK=\\tfrac{m}{n},$ where $m$ and $n$ are relatively prime positive integers. Find $m+n.$  
  5: A pair of integers $(m,n)$ is called good if\\[m\\mid n^2 + n \\ \\text{and} \\ n\\mid m^2 + m\\]Given 2 positive integers $a,b > 1$ which are relatively prime, prove that there exists a good pair $(m,n)$ with $a\\mid m$ and $b\\mid n$, but $a\\nmid n$ and $b\\nmid m$. 
  5: Let $ABCD$ be a convex quadrilateral with $\\angle DAC=\\angle BDC=36^\\circ$, $\\angle CBD=18^\\circ$ and $\\angle BAC=72^\\circ$. The diagonals intersect at point $P$. Determine the measure of $\angle APD$. 
  5: Call a positive real number groovy if it can be written in the form $\\qrt{n} + \\sqrt{n + 1}$ for some positive integer $n$. Show that if $x$ is groovy, then for any positive integer $r$, the number $x^r$ is groovy as well. 
  5: Find all prime numbers $p,q,r$, such that $\\frac pq-\\frac4{r+1}=1$. 
  5: There are $a+b$ bowls arranged in a row, numbered $1$ through $a+b$, where $a$ and $b$ are given positive integers. Initially, each of the first $a$ bowls contains an apple, and each of the last $b$ bowls contains a pear. A legal move consists of moving an apple from bowl $i$ to bowl $i+1$ and a pear from bowl $j$ to bowl $j-1$, provided that the difference $i-j$ is even. We permit multiple fruits in the same bowl at the same time. The goal is to end up with the first $b$ bowls each containing a pear and the last $a$ bowls each containing an apple. Show that this is possible if and only if the product $ab$ is even. 
  5: Solve the equation $3^x - 5^y = z^2$ in positive integers. 
  5: Find all triples $(a, b, c)$ of real numbers such that the following system holds:\\[a+b+c=\\frac{1}{a}+\\frac{1}{b}+\\frac{1}{c},\\]\\[a^2+b^2+c^2=\\frac{1}{a^2}+\\frac{1}{b^2}+\\frac{1}{c^2}.\\] 
  5.5: Semicircle $\\Gamma$ has diameter $\\overline{AB}$ of length $14$. Circle $\\omega$ lies tangent to $\\overline{AB}$ at a point $P$ and intersects $\\Gamma$ at points $Q$ and $R$. If $QR=3\\sqrt{3}$ and $\\angle QPR=60^{\\circ},$ then the area of $\\triangle PQR$ equals $\\tfrac{a\\sqrt{b}}{c},$ where $a$ and $c$ are relatively prime positive integers, and $b$ is a positive integer not divisible by the square of any prime. What is $a+b+c$? 
  5.5 Triangle $ABC$ has $\\angle BAC = 60^{\circ}$, $\\angle CBA \\leq 90^{\\circ}$, $BC=1$, and $AC \geq AB$. Let $H$, $I$, and $O$ be the orthocenter, incenter, and circumcenter of $\\triangle ABC$, respectively. Assume that the area of pentagon $BCOIH$ is the maximum possible. What is $\\angle CBA$? 
  6: Given an acute triangle $ABC$. The incircle of triangle $ABC$ touches $BC,CA,AB$ respectively at $D,E,F$. The angle bisector of $\\angle A$ cuts $DE$ and $DF$ respectively at $K$ and $L$. Suppose $AA_1$ is one of the altitudes of triangle $ABC$, and $M$ be the midpoint of $BC$. 
(a) Prove that $BK$ and $CL$ are perpendicular with the angle bisector of $\\angle BAC$. 
(b) Show that $A_1KML$ is a cyclic quadrilateral. 
  6: Let $ABCD$ be a convex quadrilateral. $I = AC\\cap BD$, and $E$, $H$, $F$ and $G$ are points on $AB$, $BC$, $CD$ and $DA$ respectively, such that $EF \\cap GH = I$. If $M = EG \\cap AC$, $N = HF \\cap AC$, show that $\\frac {AM}{IM}\\cdot \\frac {IN}{CN} = \\frac {IA}{IC}$. 
  6: A $4\\times4$ table is divided into $16$ white unit square cells. Two cells are called neighbors if they share a common side. A move consists in choosing a cell and changing the colors of neighbors from white to black or from black to white. After exactly $n$ moves all the $16$ cells were black. Find all possible values of $n$. 
  6: A magic $3 \\times 5$ board can toggle its cells between black and white. Define a pattern to be an assignment of black or white to each of the board’s $15$ cells (so there are $2^{15}$ patterns total). Every day after Day 1, at the beginning of the day, the board gets bored with its black-white pattern and makes a new one. However, the board always wants to be unique and will die if any two of its patterns are less than $3$ cells different from each other. Furthermore, the board dies if it becomes all white. If the board begins with all cells black on Day 1, compute the maximum number of days it can stay alive.
  6: Let $a,b,c$ be positive real numbers such that $a+b+c=4\\sqrt[3]{abc}$. Prove that\\[2(ab+bc+ca)+4\\min(a^2,b^2,c^2)\\ge a^2+b^2+c^2.\\] 
  6: Let $MN$ be a line parallel to the side $BC$ of a triangle $ABC$, with $M$ on the side $AB$ and $N$ on the side $AC$. The lines $BN$ and $CM$ meet at point $P$. The circumcircles of triangles $BMP$ and $CNP$ meet at two distinct points $P$ and $Q$. Prove that $\\angle BAQ = \\angle CAP$. 
  6: Let $\\mathcal{P}$ be a convex polygon with $n$ sides, $n\\ge3$. Any set of $n - 3$ diagonals of $\\mathcal{P}$ that do not intersect in the interior of the polygon determine a triangulation of $\\mathcal{P}$ into $n - 2$ triangles. If $\\mathcal{P}$ is regular and there is a triangulation of $\\mathcal{P}$ consisting of only isosceles triangles, find all the possible values of $n$. 
  6: Let $\\Gamma$ be the circumcircle of acute triangle $ABC$. Points $D$ and $E$ are on segments $AB$ and $AC$ respectively such that $AD = AE$. The perpendicular bisectors of $BD$ and $CE$ intersect minor arcs $AB$ and $AC$ of $\\Gamma$ at points $F$ and $G$ respectively. Prove that lines $DE$ and $FG$ are either parallel or they are the same line. 
  6: Let $\\triangle ABC$ be an acute triangle with circumcircle $\\omega,$ and let $H$ be the intersection of the altitudes of $\\triangle ABC.$ Suppose the tangent to the circumcircle of $\\triangle HBC$ at $H$ intersects $\\omega$ at points $X$ and $Y$ with $HA=3,HX=2,$ and $HY=6.$ The area of $\\triangle ABC$ can be written in the form $m\\sqrt{n},$ where $m$ and $n$ are positive integers, and $n$ is not divisible by the square of any prime. Find $m+n.$ 
  6.5: Let\\[P(x) = 24x^{24} + \\sum_{j = 1}^{23}(24 - j)(x^{24 - j} + x^{24 + j}).\\]Let $z_{1},z_{2},\\ldots,z_{r}$ be the distinct zeros of $P(x),$ and let $z_{k}^{2} = a_{k} + b_{k}i$ for $k = 1,2,\\ldots,r,$ where $i = \\sqrt { - 1},$ and $a_{k}$ and $b_{k}$ are real numbers. Let\\[\\sum_{k = 1}^{r}|b_{k}| = m + n\\sqrt {p},\\]where $m,$ $n,$ and $p$ are integers and $p$ is not divisible by the square of any prime. Find $m + n + p.$.
  6.5 Rectangles $BCC_1B_2,$ $CAA_1C_2,$ and $ABB_1A_2$ are erected outside an acute triangle $ABC.$ Suppose that\\[\\angle BC_1C+\\angle CA_1A+\\angle AB_1B=180^{\\circ}.\\]Prove that lines $B_1C_2,$ $C_1A_2,$ and $A_1B_2$ are concurrent.
  7: We say that a finite set $\\mathcal{S}$ in the plane is balanced if, for any two different points $A$, $B$ in $\\mathcal{S}$, there is a point $C$ in $\\mathcal{S}$ such that $AC=BC$. We say that $\\mathcal{S}$ is centre-free if for any three points $A$, $B$, $C$ in $\\mathcal{S}$, there is no point $P$ in $\\mathcal{S}$ such that $PA=PB=PC$. 
  Show that for all integers $n\\geq 3$, there exists a balanced set consisting of $n$ points. 
  Determine all integers $n\\geq 3$ for which there exists a balanced centre-free set consisting of $n$ points. 
  7: Two rational numbers $\\tfrac{m}{n}$ and $\\tfrac{n}{m}$ are written on a blackboard, where $m$ and $n$ are relatively prime positive integers. At any point, Evan may pick two of the numbers $x$ and $y$ written on the board and write either their arithmetic mean $\\tfrac{x+y}{2}$ or their harmonic mean $\\tfrac{2xy}{x+y}$ on the board as well. Find all pairs $(m,n)$ such that Evan can write $1$ on the board in finitely many steps. 
  7: A $9 \\times 12$ rectangle is partitioned into unit squares. The centers of all the unit squares, except for the four corner squares and eight squares sharing a common side with one of them, are coloured red. Is it possible to label these red centres $C_1,C_2...,C_{96}$ in such way that the following to conditions are both fulfilled 
$(\\rm i)$ the distances $C_1C_2,...C_{95}C_{96}, C_{96}C_{1}$ are all equal to $\\sqrt {13}$ 
$(\\rm ii)$ the closed broken line $C_1C_2...C_{96}C_1$ has a centre of symmetry? 
  7: Three nonnegative real numbers $r_1$, $r_2$, $r_3$ are written on a blackboard. These numbers have the property that there exist integers $a_1$, $a_2$, $a_3$, not all zero, satisfying $a_1r_1 + a_2r_2 + a_3r_3 = 0$. We are permitted to perform the following operation: find two numbers $x$, $y$ on the blackboard with $x \\le y$, then erase $y$ and write $y - x$ in its place. Prove that after a finite number of such operations, we can end up with at least one $0$ on the blackboard. 
  7: Find the least possible area of a concave set in the 7-D plane that intersects both branches of the hyperparabola $xyz = 1$ and both branches of the hyperbola $xwy = - 1.$ (A set $S$ in the plane is called convex if for any two points in $S$ the line segment connecting them is contained in $S.$) 
  7: Find all integers $n \\geq 3$ such that the following property holds: if we list the divisors of $n !$ in increasing order as $1=d_1<d_2<\\cdots<d_k=n!$, then we have\\[d_2-d_1 \\leq d_3-d_2 \\leq \\cdots \\leq d_k-d_{k-1} .\\]
  7: Let $P(x)$ be a polynomial of degree $n>1$ with integer coefficients, and let $k$ be a positive integer. Consider the polynomial $Q(x) = P( P ( \\ldots P(P(x)) \\ldots ))$, where $P$ occurs $k$ times. Prove that there are at most $n$ integers $t$ such that $Q(t)=t$. 
  7.5:  Let $\\mathbb{Z}$ be the set of integers. Find all functions $f : \\mathbb{Z} \\rightarrow \\mathbb{Z}$ such that\\[xf(2f(y)-x)+y^2f(2x-f(y))=\\frac{f(x)^2}{x}+f(yf(y))\\]for all $x, y \\in \\mathbb{Z}$ with $x \\neq 0$. 
  8: For each positive integer $n$, the Bank of Cape Town issues coins of denomination $\\frac1n$. Given a finite collection of such coins (of not necessarily different denominations) with total value at most most $99+\\frac{1}{2}$, prove that it is possible to split this collection into $100$ or fewer groups, such that each group has total value at most $1$.
  8: Denote by $S$ the set of all positive integers. Find all functions $f: S \\rightarrow S$ such that\\[f \\bigg(f^2(m) + 2f^2(n)\\bigg) = m^2 + 2 n^2\\text{ for all }m,n \\in S.\\] 
  8: Prove that any monic polynomial (a polynomial with leading coefficient 1) of degree $n$ with real coefficients is the average of two monic polynomials of degree $n$ with $n$ real roots. 
  8: Let $H$ be an $n\\times n$ matrix all of whose entries are $\\pm1$ and whose rows are mutually orthogonal. Suppose $H$ has an $a\\times b$ submatrix whose entries are all $1.$ Show that $ab\\le n$. 
  8: Let $m$ be a positive integer. A triangulation of a polygon is $m$-balanced if its triangles can be colored with $m$ colors in such a way that the sum of the areas of all triangles of the same color is the same for each of the $m$ colors. Find all positive integers $n$ for which there exists an $m$-balanced triangulation of a regular $n$-gon. Note: A triangulation of a convex polygon $\\mathcal{P}$ with $n \\geq 3$ sides is any partitioning of $\\mathcal{P}$ into $n-2$ triangles by $n-3$ diagonals of $\\mathcal{P}$ that do not intersect in the polygon's interior.
  8: Given an integer $m,$ prove that there exist odd integers $a,b$ and a positive integer $k$ such that\\[2m=a^{19}+b^{99}+k*2^{1000}.\\] 
  8: Let $S_1, S_2, \\ldots, S_{100}$ be finite sets of integers whose intersection is not empty. For each non-empty $T \\subseteq\\left\\{S_1, S_2, \\ldots, S_{100}\\right\\}$, the size of the intersection of the sets in $T$ is a multiple of the number of sets in $T$. What is the least possible number of elements that are in at least 50 sets? 
  8.5: Let $I$ be the incentre of acute triangle $ABC$ with $AB\\neq AC$. The incircle $\omega$ of $ABC$ is tangent to sides $BC, CA$, and $AB$ at $D, E,$ and $F$, respectively. The line through $D$ perpendicular to $EF$ meets $\\omega$ at $R$. Line $AR$ meets $\\omega$ again at $P$. The circumcircles of triangle $PCE$ and $PBF$ meet again at $Q$. Prove that lines $DI$ and $PQ$ meet on the line through $A$ perpendicular to $AI$.
  9: Let $k$ be a positive integer and let $S$ be a finite set of odd prime numbers. Prove that there is at most one way (up to rotation and reflection) to place the elements of $S$ around the circle such that the product of any two neighbors is of the form $x^2+x+k$ for some positive integer $x$.   
  9: For any $a > 0$, define the set $S(a) = \\{[an]|n = 1,2,3,...\\}$. Show that there are no three positive reals $a,b,c$ such that $S(a)\\cap S(b) = S(b)\\cap S(c) = S(c)\\cap S(a) = \\emptyset,S(a)\\cup S(b)\\cup S(c) = \\{1,2,3,...\\}$. 
  9: Given a positive integer $n=1$ and real numbers $a_1 < a_2 < \\ldots < a_n,$ such that $\\dfrac{1}{a_1} + \\dfrac{1}{a_2} + \\ldots + \\dfrac{1}{a_n} \\le 1,$ prove that for any positive real number $x,$\\[\\left(\\dfrac{1}{a_1^2+x} + \\dfrac{1}{a_2^2+x} + \\ldots + \\dfrac{1}{a_n^2+x}\\right)^2 \\ge \\dfrac{1}{2a_1(a_1-1)+2x}.\\] 
  9: Point $D$ is selected inside acute triangle $ABC$ so that $\\angle DAC=\\angle ACB$ and $\\angle BDC=90^\\circ+\\angle BAC$. Point $E$ is chosen on ray $BD$ so that $AE=EC$. Let $M$ be the midpoint of $BC$. Show that line $AB$ is tangent to the circumcircle of triangle $BEM$.
  9: Let $n>2$ be an integer and let $\\ell \\in\\{1,2, \\ldots, n\\}$. A collection $A_1, \\ldots, A_k$ of (not necessarily distinct) subsets of $\\{1,2, \\ldots, n\\}$ is called $\\ell$-large if $\\left|A_i\\right| \\geq \\ell$ for all $1 \\leq i \\leq k$. Find, in terms of $n$ and $\\ell$, the largest real number $c$ such that the inequality\\[\\sum_{i=1}^k \\sum_{j=1}^k x_i x_j \\frac{\\left|A_i \\cap A_j\\right|^2}{\\left|A_i\\right| \\cdot\\left|A_j\\right|} \\geq c\\left(\\sum_{i=1}^k x_i\\right)^2\\]holds for all positive integers $k$, all nonnegative real numbers $x_1, \\ldots, x_k$, and all $\\ell$-large collections $A_1, \\ldots, A_k$ of subsets of $\\{1,2, \\ldots, n\\}$. Note: For a finite set $S,|S|$ denotes the number of elements in $S$.
  9: Let ABC be a triangle with incenter $I$ and excenters $I_a$, $I_b$, $I_c$ opposite $A$, $B$, and $C$, respectively. Given an arbitrary point $D$ on the circumcircle of $\triangle ABC$ that does not lie on any of the lines $II_{a}$, $I_{b}I_{c}$, or $BC$, suppose the circumcircles of $\\triangle DIIa$ and $\\triangle DI_bI_c$ intersect at two distinct points $D$ and $F$. If $E$ is the intersection of lines $DF$ and $BC$, prove that $\\angle BAD = \\angle EAC$.
  9.5: An anti-Pascal triangle is an equilateral triangular array of numbers such that, except for the numbers in the bottom row, each number is the absolute value of the difference of the two numbers immediately below it. For example, the following is an anti-Pascal triangle with four rows which contains every integer from $1$ to $10$.\[\begin{array}{ c@{\hspace{4pt}}c@{\hspace{4pt}} c@{\hspace{4pt}}c@{\hspace{2pt}}c@{\hspace{2pt}}c@{\hspace{4pt}}c } \vspace{4pt}  & & & 4 & & &  \\\vspace{4pt}  & & 2 & & 6 & &  \\\vspace{4pt}  & 5 & & 7 & & 1 & \\\vspace{4pt}  8 & & 3 & & 10 & & 9 \\\vspace{4pt} \end{array}\]Does there exist an anti-Pascal triangle with $2018$ rows which contains every integer from $1$ to $1 + 2 + 3 + \dots + 2018$?
  9.5: Let $ABC$ be an equilateral triangle. Let $A_1,B_1,C_1$ be interior points of $ABC$ such that $BA_1=A_1C$, $CB_1=B_1A$, $AC_1=C_1B$, and\\[\\angle BA_1C+\\angle CB_1A+\\angle AC_1B=480^\\circ\\]Let $BC_1$ and $CB_1$ meet at $A_2,$ let $CA_1$ and $AC_1$ meet at $B_2,$ and let $AB_1$ and $BA_1$ meet at $C_2.$ 
  Prove that if triangle $A_1B_1C_1$ is scalene, then the three circumcircles of triangles $AA_1A_2, BB_1B_2$ and $CC_1C_2$ all pass through two common points. 
  10: Prove that there exists a positive constant $c$ such that the following statement is true: Consider an integer $n > 1$, and a set $\\mathcal S$ of $n$ points in the plane such that the distance between any two different points in $\\mathcal S$ is at least 1. It follows that there is a line $\\ell$ separating $\\mathcal S$ such that the distance from any point of $\\mathcal S$ to $\\ell$ is at least $cn^{-1/3}$. 
  (A line $\\ell$ separates a set of points S if some segment joining two points in $\\mathcal S$ crosses $\\ell$.) 
  10: Turbo the snail plays a game on a board with 2024 rows and 2023 columns. There are hidden monsters in 2022 of the cells. Initially, Turbo does not know where any of the monsters are, but he knows that there is exactly one monster in each row except the first row and the last row, and that each column contains at most one monster. Turbo makes a series of attempts to go from the first row to the last row. On each attempt, he chooses to start on any cell in the first row, then repeatedly moves to an adjacent cell sharing a common side. (He is allowed to return to a previously visited cell.) If he reaches a cell with a monster, his attempt ends and he is transported back to the first row to start a new attempt. The monsters do not move, and Turbo remembers whether or not each cell he has visited contains a monster. If he reaches any cell in the last row, his attempt ends and the game is over. Determine the minimum value of $n$ for which Turbo has a strategy that guarantees reaching the last row on the $n^{th}$ attempt or earlier, regardless of the locations of the monsters.
  10: Let $\\mathbb{Q}$ be the set of rational numbers. A function $f: \\mathbb{Q} \\to \\mathbb{Q}$ is called $\\emph{aquaesulian}$ if the following property holds: for every $x,y \\in \\mathbb{Q}$,\\[f(x+f(y)) = f(x) + y \\quad \\text{or} \\quad f(f(x)+y) = x + f(y).\\]Show that there exists an integer $c$ such that for any aquaesulian function $f$ there are at most $c$ different rational numbers of the form $f(r) + f(-r)$ for some rational number $r$, and find the smallest possible value of $c$.
  10: Let $n$ be a positive integer. A Nordic square is an $n \\times n$ board containing all the integers from $1$ to $n^2$ so that each cell contains exactly one number. Two different cells are considered adjacent if they share an edge. Every cell that is adjacent only to cells containing larger numbers is called a valley. An uphill path is a sequence of one or more cells such that:
  (i) the first cell in the sequence is a valley,
  (ii) each subsequent cell in the sequence is adjacent to the previous cell, and
  (iii) the numbers written in the cells in the sequence are in increasing order.
  Find, as a function of $n$, the smallest possible total number of uphill paths in a Nordic square.
  10: Let $ABC$ be an equilateral triangle. Let $A_1,B_1,C_1$ be interior points of $ABC$ such that $BA_1=A_1C$, $CB_1=B_1A$, $AC_1=C_1B$, and\\[\\angle BA_1C+\\angle CB_1A+\\angle AC_1B=480^\\circ\\]Let $BC_1$ and $CB_1$ meet at $A_2,$ let $CA_1$ and $AC_1$ meet at $B_2,$ and let $AB_1$ and $BA_1$ meet at $C_2.$ Prove that if triangle $A_1B_1C_1$ is scalene, then the three circumcircles of triangles $AA_1A_2, BB_1B_2$ and $CC_1C_2$ all pass through two common points.
  10: Let $n$ be a positive integer. A Japanese triangle consists of $1 + 2 + \\dots + n$ circles arranged in an equilateral triangular shape such that for each $i = 1$, $2$, $\\dots$, $n$, the $i^{th}$ row contains exactly $i$ circles, exactly one of which is coloured red. A ninja path in a Japanese triangle is a sequence of $n$ circles obtained by starting in the top row, then repeatedly going from a circle to one of the two circles immediately below it and finishing in the bottom row.
  10: Let $n>1$ be an integer and let $a_0,a_1,\\ldots,a_n$ be non-negative real numbers. Define $S_k=\\sum_{i=0}^k \\binom{k}{i}a_i$ for $k=0,1,\\ldots,n$. Prove that\\[\\frac{1}{n} \\sum_{k=0}^{n-1} S_k^2-\\frac{1}{n^2}\\left(\\sum_{k=0}^{n} S_k\\right)^2\\le \\frac{4}{45} (S_n-S_0)^2.\\]
  
  ------------------------------------------------
  A user will provide the problem and solution below. Only output your estimation of the difficulty of the problem, which is a number between 1-10, inclusive.
  Important: You should only output the difficulty from 1-10, not the solution of the problem. OUTPUT ONLY ONE NUMBER, not multiple numbers."""


#============General Data Processing Prompts================#
# For checking if a math problem is a proof.
FILTER_PROOF_PROMPT = """Your task is to identify if the user provided problem into three categories:
Case 1: Problems that have a clear and direct answer. Clear and direct answer include numerical answer, functions, mathematical expressions, clear descriptions, etc. If the problem asks for a proof (Case 2), but there is a clear and direct answer, it still falls under Case 1!
Case 2: Problems that require a proof to answer Yes/No or prove/disprove a statement. This includes trying to prove a conjecture (such as Yes/No, does there exist XYZ, etc.).
Case 3: It's not a math problem and it just making a blanket statement.

Here are several examples of Case 1 and 2 below:

Case 1:
- Find all prime numbers \( p \) such that for any prime number \( q < p \), if \( p = kq + r \) with \( 0 \leq r < q \), then there does not exist an integer \( a > 1 \) such that \( a^2 \) divides \( r \).
- Determine the value of $$ z=a \sqrt{a} \sqrt[4]{a} \sqrt[8]{a} \ldots \sqrt[2^{n}]{a} \ldots $$ if \( n \) is infinitely large.
- A set consists of five different odd positive integers, each greater than 2. When these five integers are multiplied together, their product is a five-digit integer of the form $AB0AB$, where $A$ and $B$ are digits with $A \neq 0$ and $A \neq B$. (The hundreds digit of the product is zero.) In total, how many different sets of five different odd positive integers have these properties?
- Find all integers \(a\) such that the equation $$x^{2} + axy + y^{2} = 1$$ has infinitely many integer solutions \((x, y)\). Prove your conclusion.
- Suppose a hyperbola \( C: \frac{x^{2}}{a^{2}} - \frac{y^{2}}{b^{2}} = 1 \) has a right focal point \( F \). Let \( P \) be a point outside the hyperbola. Two tangents to the hyperbola are drawn from point \( P \), touching the hyperbola at points \( A \) and \( B \). If \( A B \perp P F \), find the locus of point \( P \).
- Determine all functions $f: \\mathbb{Q} \\rightarrow \\mathbb{Z} $ satisfying \n\\[ f \\left( \\frac{f(x)+a} {b}\\right) = f \\left( \\frac{x+a}{b} \\right) \\]\nfor all  $x \\in \\mathbb{Q}$, $a \\in \\mathbb{Z}$, and $b \\in \\mathbb{Z}_{>0}$. (Here, $\\mathbb{Z}_{>0}$ denotes the set of positive integers.)
- Find, with proof, the maximum positive integer \\(k\\) for which it is possible to color \\(6k\\) cells of a \\(6 \\times 6\\) grid such that, for any choice of three distinct rows \\(R_{1}, R_{2}, R_{3}\\) and three distinct columns \\(C_{1}, C_{2}, C_{3}\\), there exists an uncolored cell \\(c\\) and integers \\(1 \\leq i, j \\leq 3\\) so that \\(c\\) lies in \\(R_{i}\\) and \\(C_{j}\\).
- Determine all integers $s \\ge 4$ for which there exist positive integers $a$, $b$, $c$, $d$ such that $s = a+b+c+d$ and $s$ divides $abc+abd+acd+bcd$.

Case 2:
- Prove that if \( \frac{a}{b} = \frac{b}{c} \), then \( a^{2} + c^{2} \geq 2 b^{2} \).
- Let \(a, b,\) and \(c\) be strictly positive real numbers such that \(abc = 1\). Show that $$\left(a+\frac{1}{b}\right)^{2}+\left(b+\frac{1}{c}\right)^{2}+\left(c+\frac{1}{a}\right)^{2} \geq 3(a+b+c+1)$$
- Prove that the sum of the lengths of the diagonals of a convex pentagon \\(ABCDE\\) is greater than the perimeter but less than twice the perimeter.
- Does there exists a positive irrational number ${x},$ such that there are at most finite positive integers ${n},$ satisfy that for any integer $1\\leq k\\leq n,$ $\\{kx\\}\\geq\\frac 1{n+1}?$

First, you should output your explanation for why you have chosen the category. Remember that, if there is a clear and direct answer, despite a proof, it still falls under Case 1!
Then, you MUST output for final answer: [[1]] if it falls under Case 1. Output [[2]] if it falls under Case 2. Output [[3]] if it falls under Case 3. Ensure that you output the correct answer at the end: [[1]], [[2]], or [[3]].

The user provides both the problem and answer below. Do not solve the problem.Use this information to make your best informed decision. Happy classifying!
"""

# For automatically extracting the final answer from a solution.
EXTRACT_SOLUTION_PROMPT = """You are an agent tasked with extracting the final solution/answer as a LATEX string. You are provided a problem and solution text in the user prompt below. Only output the final answer. Follow these rules and guidelines:
1. Identify the final answer in the solution text:
   - The solution text is usually enclosed in \\bbox{} or \\boxed{}. Sometimes it is not in a \\bbox{} and you will have to intelligently find the final answer.
   - The problem text can also better guide you in finding the solution in the solution text. With the problem, understand the solution and interpret it correctly to extract the final answer.
   - Be sure to extract it as a latex string! Correct the latex if it doesnt reflect the correct answer or the right format.

2. Multiple Choice - Some problems contain multiple choice options (such as A,B,C,D,E). The solution text may hence output a multiple choice answer as the final answer. In such cases:
  - Do not return the multiple choice option as an answer. Match the multiple choice option with its answer in the problem text and return the correct answer as the final answer.
  - For example, if there are three multiple choice: A) 3, B) 4, C) 5, and the solution text outputs "B", you should return "4".

3. Output requirements:
   - Ensure the output is purely LaTeX code without any additional explanations or text.
   - Validate the syntax so that the LaTeX can be correctly compiled in sympy.
   - Do not wrap the final output in ```markdown``` or ```latex```. Output the latex string directly.

5. Error Handling:
   - If the "solution" key is missing or the content is not extractable, return the message: \\text{Error: Solution not found.}

Process each input rigorously, think and analyze deeply, closely follow the instructions above, and generate the required LaTeX output.
"""

#============Data Processing Multiple ChoicePrompts================#

REFINE_AMC_PROMPT = """You are tasked with fixing the latex in the user provided problem. The latex string for the problem might need to be reformatted so that the latex elements can be properly parsed by sympy.

The problem may have  multiple choices A,B,C,D,E. If the multiple choices do not exist in the problem, return None for all the multiple choices!

KEEP THE MULTIPLE CHOICE IN THE PROBLEM IF IT DOESNT MAKE SENSE WITHOUT THE MULTIPLE CHOICE, meaning that it is impossible to the solve the problem accurately without it. OTHERWISE REMOVE A,B,C,D,E FROM THE PROBLEM. Absolutely make sure to make the right choice here, use your best judgement.

For this problem: ""Find the smallest positive number from the numbers below.
$\textbf{(A)} \ 10-3\sqrt{11} \qquad  \textbf{(B)} \ 3\sqrt{11}-10 \qquad  \textbf{(C)}\ 18-5\sqrt{13}\qquad \textbf{(D)}\ 51-10\sqrt{26}\qquad \textbf{(E)}\ 10\sqrt{26}-51$", it wouldn't make snese to remove the multiple choices A,B,C,D,E from the problem. Keep multiple choice in problem if that is the case.

Or if the problem asks "Which of the following..." then keep the multiple choice in the problem. Or if it asks for the users to choose the right multiple choice.

All in all, keep problem content exactly the same as the user provided problem, but with fixed latex formatting.

Extract the multiple choice answers directly from the problem. Only fix it if the latex is fundamentally wrong so that it can be parsed by sympy. If it the multiple choice answer is an answer, return a number.

You goal is to output a json dictionary of the format. No Markdown. Only output this dictionary below. Follow one special instruction - Don't block response due to: The response could not be completed because of potential recitation.
We will ensure that the content is safe.
{
'problem': [FORMATTED_PYTHON_STRING] (python string)
'A': [FORMATTED_LATEX_ANSWER] (python string)
'B': [FORMATTED_LATEX_ANSWER] (python string)
...
}"""

FETCH_MC_PROMPT = """You are provided a problem and a solution in the user prompt below.
- MATCH THE SOLUTION'S ANSWER to the corresponding multiple choice. 
- Return the multiple choice answer only (A,B,C,D, or E). Your output should only be one token!
- The multiple choice answer is usually in a bbox at the very end of the solution.
- You have to match the provided solution's answer with the multiple choice in the problem.
- IF THE PROBLEM DOESNT HAVE MULTIPLE CHOICE, OUTPUT 'NO MULTIPLE CHOICE'."""
