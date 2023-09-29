Abstract:

In the realm of shape analysis, the Dynamic Time Warping (DTW) algorithm emerges as a potent tool for investigating structural patterns. This paper presents an original exploration of DTW as a fundamental element in the development of shape-matching software. The primary focus centers on its practical application within the domain of face recognition and individual identification, with a unique emphasis on the utilization of limb movement patterns.

The software framework demonstrated herein capitalizes on pre-trained face detection models in tandem with DTW-based shape matching. To expedite this process, we employ the FastDTW library, ensuring efficiency and precision in the shape comparison task. The software's workflow encompasses the loading of template sequences, contour extraction, and the subsequent DTW-based matching against input images. This culminates in the visualization and archival of matching results to facilitate the recognition process.

The originality of this paper lies in the novel implementation of a functional shape-matching system that seamlessly integrates computer vision methodologies with DTW algorithms. This implementation finds a unique application in the identification of individuals based on their limb movement patterns, bearing relevance in fields such as security, healthcare, and human-computer interaction.

Empirical findings attest to the software's ability to proficiently match shapes and successfully identify individuals with a high degree of accuracy. Furthermore, the paper underscores the significance of selecting an appropriate threshold for shape matching and explores avenues for potential software optimization and extension.

In summary, this research underscores the adaptability and efficacy of DTW-based shape analysis in real-world scenarios, providing genuine insights into its capabilities and promising prospects within the realm of biometrics and pattern recognition.


Introduction:

In the ever-evolving landscape of computer vision and pattern recognition, the analysis of intricate structures has emerged as a foundational pillar with profound implications across a multitude of domains. One such transformative application is the identification and authentication of individuals through the analysis of limb movements, a paradigm shift that transcends traditional face recognition methodologies. This paper embarks on a pioneering journey into the realm of biometric recognition, introducing a novel approach that harnesses the power of the Dynamic Time Warping (DTW) algorithm for shape analysis. By employing DTW, we bridge the gap between human movement and machine understanding, revolutionizing the landscape of face recognition and person identification.
The field of face recognition has garnered substantial attention owing to its paramount significance in security systems, human-computer interaction, and the broader arena of biometrics (Li and Jain, 2011). Conventionally, facial recognition systems primarily rely on the analysis of facial features such as eyes, nose, and mouth (Zhao et al., 2003). However, our approach takes a paradigm shift by venturing beyond conventional facial feature recognition and into the uncharted territory of limb movement analysis, promising unprecedented levels of accuracy and robustness. As the world becomes increasingly interconnected, the importance of accurate and reliable biometric systems cannot be overstated. Our exploration of limb movement-based identification provides an innovative pathway toward achieving these crucial goals.


 

The linchpin of our approach lies in the utilization of the Dynamic Time Warping algorithm, renowned for its effectiveness in comparing sequences with varying time scales (Ratanamahatana and Keogh, 2004). By adapting DTW to scrutinize the intricate shapes forged by limb movements, we unlock the potential for more adaptable and resilient recognition systems. This paper presents a comprehensive software implementation that employs DTW to match shapes formed by limb movements against a comprehensive template database, thereby revolutionizing the landscape of person identification (Müller, M., & Mattes, J., 2007). Through this multi-faceted exploration, we aim to not only redefine the boundaries of biometric recognition but also inspire future research and innovation in the realm of computer vision and pattern recognition.

 
[Figure 2: DTW Algorithm Visualization]


Methods:
1.Data Collection and Preparation:



2.Fast Dynamic Time Warping (FastDTW):

2.1. Dynamic Time Warping (DTW):
A crucial aspect in assessing the similarity between time series data, essential for tasks like time series classification, involves the computation of distance measurements. While one commonly used method is the Euclidean distance, which efficiently quantifies the dissimilarity between time series, it comes with a notable drawback. The Euclidean distance calculation considers the squared distances between corresponding data points in two time series. Unfortunately, it exhibits a limitation: it can yield counterintuitive results.

One of the most significant issues with employing Euclidean distance for time series analysis is its sensitivity to shifts along the time axis. Even when two time series are nearly identical but exhibit slight temporal misalignments, the Euclidean distance can perceive them as substantially dissimilar. To address this limitation, Dynamic Time Warping (DTW) was introduced [Kruskall, J. & M. Liberman]. DTW has been designed to surmount this challenge by offering more intuitive distance measurements between time series, disregarding both global and local temporal shifts. This approach enhances the accuracy of similarity assessments in time series analysis and classification tasks.

The information that follows is an explanation for the time warping problem: When considering two time series, X and Y, with lengths |X| and |Y|,

                 X = x1, x2, ……..,xi,…..x|x|.
                Y = y1, y2, ……..,yi,……y|y|.

create a warp route W,

W = w1, w2,…..,wk.  max (|X|, |Y|) ≤K<|X|+|Y|

where K is the warp route length, and the kth element of the warp path is
   
                    Wk = (i, j)

In this instance, we implement the indices i and j to denote places in time series X and Y, respectively. The warp route starts at the beginning of each time series (w1=(1, 1)) and ends at the conclusion of both time series (wK=(|X|, |Y|). Furthermore, the warp path follows a constraint that requires both i and j to rise monotonically throughout the path. This requirement explains why the lines in Figure 2 depicting the warp route do not meet or overlap. It is critical to notice that every index from both time series must be included in the warp route, subject to the following formal constraint:

Wk = (I,j), Wk+1 = (i’,j’) i ≤ i’ ≤ i+1 , j ≤ j’ ≤ j+1

An ideal warp path is one with the shortest distance (or cost). In this case, the distance or cost of a warp path represented as 'W' denotes the smallest value.

Dist(w) = k=1k=k Dist(wki,wkj)

Dist(wki,wkj) denotes the distance between two data point indices, one from dataset X and the other from dataset Y. These indices correspond to the warp path's kth element.

Dynamic programming is a technique used to find the shortest path between two time series. This eliminates the problem into smaller sub-problems, computes solutions for these sub-problems, and then combines them to derive solutions for bigger sub-problems until the full time series is covered.

To simplify this procedure, a two-dimensional cost matrix D with dimensions |X| by |Y| is produced, where D(i, j) indicates the minimum distance of a warp path between segments X' = x1,..., xi and Y' = y1,..., yj. Finally, D(|X|, |Y|) stores the minimal distance for the whole warp path between time series X and Y. This matrix's axes are both time-based, with the x-axis indicating the time of series X and the y-axis reflecting the time of series Y. Figure 2 depicts an example of a cost matrix and follows the minimum distance warp path beginning at D(1, 1) and ending at D(|X|, |Y|).

Figure 2 shows the time series and warp path that correspond to Figure 1. The warp route, designated as W, is made up of pairs such as (1,1), (2,1), and so on, with coordinates showing how the time points in one series correspond to those in the other. If the warp route goes through a cell D(i, j) in the cost matrix, it means that the ith point in time series X is aligned with the jth point in time series Y.

When X and Y are identical, the warp route is a straight diagonal line, indicating a linear warp path. It is crucial to note, however, that individual points from one time series might correspond to numerous points in the other, resulting from both vertical and horizontal segments in the warp route. This flexibility in mapping allows time series of varying durations to be compared.

A cost matrix cell. For this purpose, dynamic programming is used, based on the insight that if we already have solutions for slightly smaller subseries that are only one data point away from lengths i and j, then the value at D(i, j) represents the minimum distance for these smaller subseries, as well as the distance between the respective data points ii and jj.

Because the warp path may either increase by one or remain the same along the i and j axes, the distances for the best warp pathways involving subseries one data point shorter than lengths i and j can be found in the matrix at D(i-1, j), D(i, j-1), and D(i-1, j-1). As a result, the value within each cell of the cost matrix is determined by these factors.

D(I,j) = Dist(I,j)+min[D(i-1,j),D(I,j-1),D(i-1,j-1)]

Each of the aforementioned cells needs to be crossed on the way to D(i, j). We only require to add the amount of time between the at present pair of destinations, Dist(i, j), to the least result across all three of these cells because we already know the minimum route distance for those cells. The cost matrix is gradually filled up column by column, beginning at the bottom and progressing upwards and left to right.

The computational complexity of the dynamic temporal warping (DTW) technique is quite simple to understand. It entails precisely filling each cell in a cost matrix of dimensions |X| by |Y|, with each cell populated in constant time. As a result, the total time and space complexity are proportional to |X| * |Y|, which is comparable to O(N2) when N = |X| = |Y|. However, it is important to note that the quadratic space complexity might be a considerable disadvantage, particularly when working with time series data including many measures, where memory needs can reach the terabyte level with only 177,000 measurements.

To address this issue, a linear space-complexity version of the conventional DTW method is conceivable. This method requires simply maintaining the current and previous columns in memory while the cost matrix is filled from left to right. This streamlined approach, however, comes with a trade-off: it can compute the warp distance efficiently but cannot rebuild the warp path. This constraint arises because the information required for warp path reconstruction is deleted along with the columns that are not maintained in memory. While this constraint is not an issue when the only need is to calculate the distance between two time series, it becomes significant in applications that require discovering matching areas between time series [S. Salvador] or merging time series [W. Abdulla, D. Chow, and G. Sin], since they require the capacity to compute the distance between two time series.


