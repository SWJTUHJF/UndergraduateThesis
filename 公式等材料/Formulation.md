Notation

|        Variable        | Annotation                                                   |
| :--------------------: | :----------------------------------------------------------- |
|          $A$           | Set of arcs                                                  |
|          $R$           | Set of user's origins                                        |
|          $S$           | Set of user's destinations                                   |
|          $u$           | Maximal number of path provided to each user                 |
|          $l$           | Minimal number of path provided to each user                 |
|     $\Pi^{r,s}_u$      | The u-shortest paths connecting OD pair $r\in R$ and $s\in S$ under UE flow pattern |
|         $x_a$          | The arc flow of arc $a\in A$                                 |
|         $t_a$          | The arc travel time of arc $a\in A$                          |
|       $n^{r,s}$        | Number of paths among provided to users between OD pair $r\in R$ and $s\in S$ |
|     $h^{r,s}_\pi$      | The flow of path $\pi\in \Pi^{rs}_u$ between OD pair $r\in R$ and $s\in S$ |
| $\delta^{r,s}_{a,\pi}$ | $\delta^{r,s}_{a,\pi}=1$ if arc $a\in A$ is included in path $\pi\in\Pi^{r,s}_u$ connecting OD pair $r\in R$ and $s\in S$;  $\delta^{r,s}_{a,\pi}=0$ otherwise |
|    $\tau^{r,s}_\pi$    | $\tau^{r,s}_\pi=0$ if user is aware of path $\pi\in\Pi^{rs}_u$ connecting OD pair $r\in R$ and $s\in S$; $\tau^{r,s}_\pi=1$ otherwise |
|       $q^{r,s}$        | The fixed and deterministic OD demand between OD pair $r\in R$ and $s\in S$ |

$$
\min F(\bold{x,\tau})=\sum_{a\in A}x_at_a(x_a)
\\
s.t.\begin{cases}
\sum\limits_{\pi\in\Pi^{r,s}_u}(1-\tau^{r,s}_\pi)\geq l \quad\forall r\in R,\forall s\in S
\\
\\
\sum\limits_{\pi\in\Pi^{r,s}_u}(1-\tau^{r,s}_\pi)\leq u \quad\forall r\in R,\forall s\in S
\end{cases}
$$

​                                                                                              where $\bold{x}=\bold{x(\tau)}$ is implicitly determined by:
$$
\min f(\bold{x,\tau})=\sum_{a\in A}\int^{\sum\limits_{r\in R}\sum\limits_{s\in S}\sum\limits_{\pi\in\Pi^{r,s}}\delta^{r,s}_{a,\pi}h^{r,s}_\pi}t_a(x)dx
\\
s.t.\begin{cases}
\sum\limits_{\pi\in\Pi^{r,s}}h^{r,s}_\pi=q^{r,s},\quad\forall r\in R,\forall s\in S
\\
\\
h^{r,s}_\pi\geq 0,\quad\forall r\in R,\forall s\in S,\forall\pi\in\Pi^{r,s}
\\
\\
\tau^{r,s}_\pi h^{r,s}_\pi=0,\quad\forall r\in R,\forall s\in S,\forall\pi\in\Pi^{r,s}
\end{cases}
$$
