# SMPR
Language R

- [Метрические алгоритмы классификации](#Метрические-алгоритмы-классификации)
  - [1NN](#1NN)
  - [KNN](#KNN)
  - [KWNN](#KWNN)
  - [Парзеновское окно (PW)](#Парзеновское-окно-pw)
  - [Потенциальные функции (PF)](#Потенциальные-функции-pf)
  - [Алгоритм STOLP](#Алгоритм-STOLP)
- [Байесовские классификаторы](#Байесовские-классификаторы)
  - [Наивный байесовский классификатор](#Наивный-байесовский-классификатор)
  - [Изолинии](#Изолинии)
  - [Plug-in алгоритм](#Plug-in-алгоритм)  
  - [Линейный дискриминант Фишера](#Линейный-дискриминант-Фишера)  
- [Линейные алгоритмы классифиикации](#Линейные-алгоритмы-классифиикации)
  
 
  
  
  <center>
<table>
  <tbody>
    <tr>
      <th>Метод</th>
      <th>Параметры</th>
      <th>Loo</th>
    </tr>
    <tr>
      <td>KWNN</a></td>
      <td>k=9</td>
      <td>0.0333</td>
    </tr>
    <tr>
      <td>KNN</a></td>
      <td>k=6</td>
      <td>0.0333</td>
    </tr>
    <tr>
      <td>Парзеновские окна</a></td>
      <td>h=0.4  ( Прямоугольное ядро )</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>Парзеновские окна</a></td>
      <td>h=0.4  ( Треугольное ядро )</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>Парзеновские окна</a></td>
      <td>h=0.4  ( ядро Епанечникова )</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>Парзеновские окна</a></td>
      <td>h=0.4  ( Квартическое ядро )</td>
      <td>0.04</td>
    </tr>
      <tr>
      <td>Парзеновские окна</a></td>
      <td>h=0.1  ( Гауссовское ядро )</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>Потенциальные функции</a></td>
      <td>h=0.4  ( Прямоугольное ядро )</td>
      <td>Подбор гамма</td>
    </tr>
       	  </tbody>
   </table>
 
 
## Метрические алгоритмы классификации

**Метрические методы обучения** -- методы, основанные на анализе сходства объектов.

Метрические алгоритмы классификации опираются на **_гипотезу компактности_**: схожим объектам соответствуют схожие ответы.

Метрические алгоритмы классификации с обучающей выборкой *Xl* относят объект *u* к тому классу *y*, для которого **суммарный вес ближайших обучающих объектов ![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29) максимален**:

![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29%20%3D%20%5Csum_%7Bi%20%3A%20y_%7Bu%7D%5E%7B%28i%29%7D%20%3D%20y%7D%20w%28i%2C%20u%29%20%5Crightarrow%20max)

где весовая функция *w(i, u)* оценивает степень важности *i*-го соседа для классификации объекта *u*.

Функция ![](https://latex.codecogs.com/gif.latex?W_y%28u%2C%20X%5El%29) называется **_оценкой близости объекта u к классу y_**. Выбирая различную весовую функцию *w(i, u)* можно получать различные метрические классификаторы.

### Методы ближайших соседей

Классификация ирисов Фишера методом 1NN (ближайшего соседа)

Решалась задача классификации. В качестве обучающей выборки была взята матрица ирисов фишера по длине и ширине лепестка.
Требовалось построить карту классификации на основе данных обучающей выборки.
В качестве метода классификации использовался метод 1NN.
## 1NN 
# Метод 1NN состоит в следующем: 
	1.Для классифицируемого объекта вычисляются расстояния от него до каждого объекта обучающей выборки.
	2.Обучающая выборка сортируется по возрастанию расстояния от каждого объекта выборки до классифицируемого
	3.Классифицируемому объекту присваивается тот же класс, что и ближайшего к нему объекта выборки.
 
 
  <p><img src="img\1nn.png" ></p>
  
Классификация ирисов Фишера методом kNN (k ближайших соседей)


Решалась задача классификации. В качестве обучающей выборки была взята матрица ирисов фишера по длине и ширине лепестка.
Требовалось построить карту классификации на основе данных обучающей выборки.
В качестве метода классификации использовался метод kNN.


  ## KNN 


 Метод kNN состоит в следующем: 
	1.Для классифицируемого объекта вычисляются расстояния от него до каждого объекта обучающей выборки
	2.Обучающая выборка сортируется по возрастанию расстояния от каждого объекта выборки до классифицируемого
	3.Подсчитывается, какой класс доминирует среди k ближайших соседей, и этот класс присваивается классифицируемому объекту
	## Алгоритм k ближайших соседей (kNN)
Алгоритм 1NN относит классифицируемый объект U к тому классу, которому принадлежит его ближайший сосед.
ὠ(i,u)=[i=1];

Алгоритм kNN относит объект к тому классу, элементов которого больше среди k ближайших соседей x(i), i=1,..,k.

Для оценки близости классифицируемого объекта *u* к классу *y* **алгоритм kNN** использует следующую функцию:

ὠ(i,u)=[i<=k], где *i* -- порядок соседа по расстоянию к классифицируемому объекту *u*, k-количество параметровю=.
**Реализация весовой функции производится следующим образом**:

``` R
distances <- matrix(NA, l, 2) # расстояния от классифицируемого объекта u до каждого i-го соседа 
for(i in 1:l) {
   distances[i, ] <- c(i, eDist(xl[i, 1:n], u))
}
orderedxl <- xl[order(distances[ , 2]), ] # сортировка расстояний
classes <- orderedxl[1:k, n + 1] # названия первых k классов (k ближайших соседей) в classes 
```

<p><img src="img\LooKnn.png" ></p>

### Преимущества:
1. Простота реализации.
2. При *k*, подобранном около оптимального, алгоритм "неплохо" классифицирует.

### Недостатки:
1. Нужно хранить всю выборку.
2. При *k = 1* неустойчивость к погрешностям (*выбросам* -- объектам, которые окружены объектами чужого класса), вследствие чего этот выброс классифицировался неверно и окружающие его объекты, для которого он окажется ближайшим, тоже.
2. При *k = l* алгоритм наоборот чрезмерно устойчив и вырождается в константу.
3. Крайне бедный набор параметров.
4. Точки, расстояние между которыми одинаково, не все будут учитываться.

## KWNN
Реализаця метода kwNN
В каждом классе выбирается
__k__ ближайших к __U__ объектов, и объект u относится к тому классу, для
которого среднее расстояние до __k__ ближайших соседей минимально.
![](http://latex.codecogs.com/gif.latex?w%28i%2Cu%29%3D%5Bi%5Cleq%20k%5Dw%28i%29)
где,
![](http://latex.codecogs.com/gif.latex?w%28i%29) — строго убывающая последовательность вещественных весов, задающая
вклад i-го соседа при классификации объекта u.

<p><img src="img\LooKwnn.png" ></p>

### Весовая функция

```
weightsKWNN = function(i, k)
{
  (k + 1 - i) / k
}
```

<center>
<table>
  <tbody>
    <tr>
      <th>Метод</th>
      <th>Параметры</th>
      <th>Точность</th>
    </tr>
    <tr>
      <td>KWNN</a></td>
      <td>k=9</td>
      <td>0.0333</td>
    </tr>
    <tr>
      <td>KNN</a></td>
      <td>k=6</td>
      <td>0.0333</td>
    </tr>
	  </tbody>
   </table>

#### Сравнение качества алгоритмов kNN и kwNN.

kNN — один из простейших алгоритмов классификации, поэтому на реальных задачах он зачастую оказывается неэффективным. Помимо точности классификации, проблемой этого классификатора является скорость классификации: если в обучающей выборке N объектов, в тестовой выборе M объектов, и размерность пространства  K, то количество операций для классификации тестовой выборки может быть оценено как O(KMN).

kwNN отличается от kNN, тем что учитывает порядок соседей классифицируемого объекта, улчшая качество классификации.

Пример, показывающий преимущество метода kwNN над kNN:
<a href="http://www.picshare.ru/view/9312480/" target="_blank"><img src="http://www.picshare.ru/uploads/181018/l7iAu3dDsZ.jpg" border="0" width="1462" height="524" title="Хостинг картинок PicShare.ru"></a>

В примере передаем параметр k=5.  Kwnn в отличии от Knn оценивает не только индекс соседа, но и  расстояние до него, из-за этого результат получается более точный, что и продемонстрировано в данном примере.


Число k выбирается методом LOO (скользящего контроля)


### Метод LOO:
1. Элемент удаляется из выборки
2. Запускается алгоритм классификации (в данном случае kNN) для полученной выборки и удалённого объекта
3. Полученный класс объекта сравнивается с реальным. В случае несовпадения классов, величина ошибки увеличивается на 1
4. Процесс повторяется для каждого объекта выборки
5. Полученная ошибка усредняется
6. Процесс повторяется для других значений k. В итоге выбирается число k с наименьшей ошибкой LOO
    

```	
##Составляем таблицу встречаемости каждого класса
counts <- table(classes)
class <- names(which.max(counts))
return (class)
}

## Метод скользящего контроля
Loo <- function(k,xl)
   {
    sum =0
    for(i in 1:dim(xl)[1]
       {
        tmpXL <- rbind(xl[1:i-1, ],
        xl[i+1:dim(xl)[1],])
        xi <- c(xl[i,1], xl[i,2])
        class <-kNN(tmpXL,xi,k)
        if(class != xl[i,3])
        sum=sum+1
       }
   sum=sum/dim(xl)[1]
   return(sum)
  }
```

 ### Преимущества:
 Преимущество LOO состоит в том, что каждый объект ровно один раз участвует в контроле, а длина обучающих подвыборок лишь на единицу меньше длины полной выборки.
 ### Недостатки:
  Недостатком LOO является большая ресурсоёмкость, так как обучаться приходится L раз. Некоторые методы обучения позволяют достаточно быстро перенастраивать внутренние параметры алгоритма при замене одного обучающего объекта другим. В этих случаях вычисление LOO удаётся заметно ускорить. 
  
  
  
  ### Парзеновское окно (PW)

Для оценки близости объекта _u_ к классу _y_ алгоритм использует следующую
функцию:

![](http://latex.codecogs.com/svg.latex?%5Clarge%20W%28i%2C%20u%29%20%3D%20K%28%5Cfrac%7B%5Crho%28u%2C%20x%5Ei_u%29%7D%7Bh%7D%29)
, где 
![](http://latex.codecogs.com/svg.latex?%5Clarge%20K%28z%29) — функция ядра.
  
  Наиболее часто используются следующие типы ядер
  
  <p><img src="img\Kernels.png" ></p>
  
 Код для реализации данных типов ядер:
```
kernelE = function(r){ return ((3/4*(1-r^2)*(abs(r)<=1)))}  #епанечникова
kernelQ = function(r){ return ((15 / 16) * (1 - r ^ 2) ^ 2 * (abs(r) <= 1))}  #квартическое
kernelT = function(r){ return ((1 - abs(r)) * (abs(r) <= 1)) }  #треугольное
kernelG = function(r){ return (((2*pi)^(-1/2)) * exp(-1/2*r^2))}  #гауссовское
kernelR = function(r){ return ((0.5 * (abs(r) <= 1) ))}  #прямоугольное
```
  __Алгоритм__ вокруг нашей классифицируемой точки _u_ строит окружность с радиусом _h_. Далее убираем точки, которые не вошли в окружность. Затем для оставшихся, считаем _weights_, суммируем  по class, и с помощью _names(which.max(weights))_ возвращаем название класса "победителя".
  
  Код алгоритма:
```
PW = function(XL,y,h,metricFunction = euclideanDistance){  
l <- dim(xl)[1]
  weights = rep(0,3)
  names(weights) = unique(xl[,3])
  for(i in 1:l)
  {
        x=XL[i,1:2]
    class=XL[i,3]
        r = metricFunction(x,y)/h
    weights[class]=kernelR(r)+weights[class];
  }
  #no p in w
     if(max(weights)==0){ return ("0") }
     else{ return (names(which.max(weights))) }
                                                        }
```
##   Карта классификации для ядра Епанечникова  и   для Треугольного ядра

<p><img src="img\ET.png" ></p>

##   Карта классификации для Квартического ядра и  для Прямоугольного ядра

<p><img src="img\KR.png" ></p>

##   Карта классификации для Гауссовского ядра


В отличии от предыдущих ядер, Гауссовское ядро устраняет проблему с классификацией точек, не попавших ни в одно окно.

<p><img src="img\G.png" ></p>


##   Loo для ядра Епанечникова   и    для Треугольного ядра

<p><img src="img\ETloo.png" ></p>

##   Loo для Квартического ядра   и   для Прямоугольного ядра

<p><img src="img\QR.png" ></p>

##   Loo для Гауссовского ядра

<p><img src="img\Gloo.png" ></p>

__Наблюдения при реализации__:
    Если _h_ сделать слишком маленьким, классифицуруется заметно меньшее количество точек. Если  же _h_ сделать слишком большим, то  учитываются точки, находящиеся на очень большом расстоянии.
   
__Плюсы:__
- прост в реализации
- хорошее качество классификации при правильно подобраном _h_
- все точки с одинаковым расстоянием будут учитаны

__Минусы:__
- необходимо хранить всю выборку целиком
- небольшой набор параметров
- диапазон параметра _h_ необходимо подбирать самостоятельно(в зависимости от плотности расположения т.)
- если ни одна точка не попала в радиус _h_, алгоритм не способен ее
классифицировать (кроме гауссовского ядра)

## Потенциальные функции (PF)

__Метод потенциальных функций__ - относится к метрическим классификаторам. В отличии от метода парзеновсуих окон, окно строится вокруг обучающих точек.
При классификации объект проверяется на близость к объектам из обучающей выборки. Простыми словами, объекты из обучающей выборки «заряжены» своим классом, а вес каждого из них при классификации зависит от «заряда» и расстояния до классифицируемого объекта.

__Алгоритм метода PF__

   - Изначально для каждого объекта выборки задаём *ширину окна* ![h_i](http://latex.codecogs.com/gif.latex?h_%7Bi%7D) эмпирически (выбирается из собственных соображений).
   - Затем для обучающих объектов вычисляем *силу потенциала* ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D). 
   - После чего каждому объекту выборки присваивается *вес* по формуле ![](http://latex.codecogs.com/gif.latex?w%28x_%7Bi%7D%2Cz%29%3D%5Cgamma_%7Bi%7DK%28%5Cfrac%7B%5Crho%28x%7Bi%7D%2Cz%29%7D%7Bh_%7Bi%7D%7D%29%3D%5Cgamma_%7Bi%7DK%28r%29), ![](http://latex.codecogs.com/gif.latex?K%28r%29) - функция ядра.
   - Суммируем веса объектов одинаковых классов. Класс "победитель" присваивается точке.


    
 __Код метода потенциальных функций:__
 
      PF = function(potentials,XL,y,h,metricFunction = euclideanDistance)
    {
     l <- nrow(XL)
     n <- ncol(XL)

     weights = rep(0,3)
    names(weights) = unique(XL[,3])
     for(i in 1:l)
    {
    
    x=XL[i,1:2]
    class=XL[i,3]
    
    r = metricFunction(x,y)/h
    weights[class] = weights[class] + potentials[i]*kernelR(r);
    }
    class = names(which.max(weights))
       #no p in w
    if (max(weights) == 0) return("0") 
      return(class)
         }
	 
Алгоритм подбора ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D):

   - Множество потенциалов ![gamma_](http://latex.codecogs.com/gif.latex?\gamma_) зануляется. Задается максимально допустимое число ошибок(eps).
   - Из обучающей выборки выбирается очередной объект ![x_i](http://latex.codecogs.com/gif.latex?x_%7Bi%7D).
   - Затем Для ![x_i](http://latex.codecogs.com/gif.latex?x_%7Bi%7D) запускаю алгоритм классификации.
   - Если  полученный класс не совпал с реальным, то *сила потенциала* для выбранного объекта увеличивается на 1. Иначе снова выбирается объект и классифицируется.
   - Алгоритм классификации с полученными значениями потенциалов запускается для каждого объекта выборки. Подсчитывается число ошибок.
   - Если число ошибок меньше заданного, то алгоритм завершает работу. Иначе снова выбирается объект из выборки.
	 
__Код алгоритма подбора ![gamma_](http://latex.codecogs.com/gif.latex?\gamma_):__

     getPotentials <- function(XL,eps,h,class) 
    {
     # get pots all elements
    l <- nrow(XL)
     n <- ncol(XL)

    potentials <- rep(0,l)
    err <- eps + 1
 
    while (err > eps) 
    {
     while (TRUE) 
     {
      # Пока не получим несоответствие классов, чтобы обновить потенциалы
      rand <- sample(1:l, 1)
     x=XL[rand,1:2]
        u <- PF(potentials,XL,x,h)

      if(colors[u] != colors[class[rand]]) {
        potentials[rand] = potentials[rand] + 1
        break
        }
     }
    # Подсчет числа ошибок
    err <- 0
    for (i in 1:l)
    {
      x = XL[i,1:2]
        points=XL[-i,1:3]
         if(colors[PF(potentials,points,x,h)]!= colors[class[i]])
     {
          err = err + 1
    }
    }
    }
     return (potentials)
    }
    
В данной программе использовал прямоугольное ядро. Алгоритм подбирает только силу потенциала , радиус потенциалов h задаются эмпирически(по собственным соображениям).

Список ненулевых потенциалов(15 номеров)

     [1] 3
     [1] 66
     [1] 71
     [1] 78
     [1] 84
     [1] 87
     [1] 94
     [1] 102
     [1] 105
     [1] 107
     [1] 119
     [1] 120
     [1] 122
     [1] 134
     [1] 139


__Задать значение ширины окна для каждого класса__
  
     SvoiH <- function(xl) 
     {
     l <- nrow(xl)
     h <- rep(0, l)
     for(i in 1:l) {
     if (xl[i, ncol(xl)] == "setosa") h[i] <- 1
     else h[i] <- 0.4
                  }
     return (h)
      }
      h <- SvoiH(xl)
      
Подбор потенциалов происходит до тех пор, пока точность алгоритма, не будет меньше заданной ошибки eps.
Иллюстрация результата работы алгоритма:
<p><img src="img\PFF2.png" ></p>

__Преимущества метода потенциальных функций:__

- Большое количество параметров для подбора. 
- Возможность использования не всей выборки

__Недостатки метода:__

- Метод непрост для понимания и алгоритмической реализации;
- Неопределённое время работы алгоритма подбора(если взять маленькое eps) ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D);
- Параметры ![gamma_i](http://latex.codecogs.com/gif.latex?%5Cgamma_%7Bi%7D). и _h_ настраиваются слишком грубо;




### Алгоритм STOLP
Алгоритм СТОЛП (STOLP) — алгоритм отбора эталонных объектов для метрического классификатора.
Отступ - величина, показывающая, насколько объект является типичным представителем своего класса. Отступ равен разности между степенью близости объекта к своему классу и суммой близостей объекта ко всем остальным классам.  
![](http://latex.codecogs.com/gif.latex?M%28x_%7Bi%7D%29%3DW_%7By_%7Bi%7D%7D%28x_%7Bi%7D%2CX%5E%7Bl%7D%29-%5Cmax%20W_%7By%7D%28x_%7Bi%7D%2C%20X%5E%7Bl%7D%29)  
Все объекты обучающей выборки можно разделить на несколько типов:

   - Эталонные объекты - наиболее типичные представители своего класса
   - Неинформативные - объекты, не влияющие значительным образом на качество классификации
   - Пограничные - объекты, имеющие отступ, близкий к нулю. Незначительное изменение в выборке, алгоритме, метрике и т.п. может повлиять на их классификацию.
   - Ошибочные - объекты с отрицательными отступами, классифицируемые неверно.
   - Шумовые объекты (выбросы) - малая группа объектов с большими отрицательными отступами. Их удаление улучшает качество классификации.

Идея алгоритма STOLP состоит в том, чтобы уменьшить исходную выборку, выбрав из неё эталонные объекты. Такой подход уменьшит размер выборки, и может улучшить качество классификации.  На вход подаётся выборка, допустимый порог ошибок и порог фильтрации выбросов. 

__Алгоритм:__ 

1. Удалить из выборки все выбросы (объекты, отступ которых меньше порога фильтрации выбросов).
2. Пересчитать все отступы заново, взять по одному эталону из каждого класса (объекты с наибольшим положительным отступом), и добавить их в множество эталонов.
3. Проклассифицировать объекты обучающей выборки, взяв в качестве обучающей выборки для этого множество эталонов. Посчитать число ошибок.
4. Если число ошибок меньше заданного порога, то алгоритм завершается.
5. Иначе присоединить ко множеству эталонов объекты с наименьшим отступом из каждого класса из числа классифицированных неправильно.
6. Повторять шаги 3-5 до тех пор, пока множество эталонов и обучающая выборка не совпадут, или не сработает проверка в пункте 4.

Реализация функции для нахождения отступа наших объектов
```
margin = function(points,classes,point,class){

  Myclass = points[which(classes==class), ]
  OtherClass = points[which(classes!=class), ]
  
  MyMargin = Parzen(Myclass,point[1:2],1,FALSE)
  OtherMargin = Parzen(OtherClass,point[1:2],1,FALSE)
  
  return(MyMargin-OtherMargin)
}

```

В итоге осталось 5 эталонных объектов, скорость работы метода после алгоритма заметно улучшилась, а точность ухудшилась незначительно. Точность алгоритма До отбора эталонных составляла: 0.96 (6 ошибок), а после: 0.94 (8 ошибок)  
<p><img src="img\STOLP.png" ></p>

## Байесовские классификаторы

<center>
<table>
  <tbody>
    <tr>
      <th>Задание</th>
      <th>Shiny</th>
    </tr>
    <tr>
      <td><a href="#Изолинии">Изолинии(Линии уровня)</a></td>
      <td><a href="https://dimitridze.shinyapps.io/line/">+</a></td>
	    </tr>
	</tbody>
	</table>

Байесовский подход опирается на теорему о том, что если плотности распределения классов известны, то алгоритм классификации,  можно выписать в явном виде(если алгоритм имеет минимальную вероятность ошибок). Для оценивания плотностей классов по выборке применяются различные подходы. В этом курсе лекций рассматриваем три: непараметрический, параметрический и оценивание смесей распределений.  

При решении задачи __классификации__, необходимо по известному вектору признаков __x__, определить класс  __y__ к которому принадлежит объект __a(x)__  по формуле : __a(x) = argmax P(y|x)__, для которого при условии x, вероятность класса y - наиболее высока. 


В случае когда __P(y__) одинаковы для всех классов, то следует выбрать класс __у__, у которого в точке __x__, плотность больше.

## Формула Байеса
<img src="https://latex.codecogs.com/gif.latex?p(y|x)=\frac{p(x,y)}{p(x)}&space;=&space;\frac{p(x|y)p(y)}{p(x)}" title="p(y|x)=\frac{p(x,y)}{p(x)} = \frac{p(x|y)p(y)}{p(x)}" /></a>
1. <img src="https://latex.codecogs.com/gif.latex?p(y|x)"/></a> - Апостериорная вероятности, т.е. вероятность того, что объект x принадлежит классу y.
2. <img src="https://latex.codecogs.com/gif.latex?p(x|y)"/></a> - функция правдободобия.
3. <img src="https://latex.codecogs.com/gif.latex?p(y)"/></a> - Априорная вероятность, т.е. вероятность появления класса.


## Наивный байесовский классификатор 
 
  Наивный байесовский классификатор – это семейство алгоритмов классификации, которые принимают одно допущение: Каждый параметр классифицируемых данных рассматривается независимо от других параметров класса.  
  Метод называется наивным т.к. предполагается, что все параметры набора данных независимы, что встречается крайне редко.  

  Обычно он используется, как эталон при сравнении различных алгоритмов классификации.  
Решающее правило принимает вид:  
![](http://latex.codecogs.com/gif.latex?a%28x%29%3Darg%20%5Cmax_%7By%5Cin%20Y%7D%28%5Cln%28%5Clambda_%7By%7DP_y%29&plus;%5Csum_%7Bj%3D1%7D%5E%7Bn%7D%5Cln%28p_%7Byj%7D%28%5Cxi_j%29%29%29)  


## Изолинии
 Случайная величина x ∈ R имеет нормальное (гауссовское) распределение с параметрами µ и σ^2, если ее плотность задается выражением:  
![f](http://simfik.ru/i/f.png)  
Параметры µ и σ^2 определяют, соответственно, мат.ожидание и дисперсию нормальной случайной величины. По центральной предельной теореме среднее арифметическое независимых случайных величин с ограниченными мат.ожиданием и дисперсией стремится к нормальному распределению. Поэтому это распределение часто используется в качестве модели шума, который определяется суммой большого количества независимых друг от друга случайных факторов.  
Собственно, <a href="https://dimitridze.shinyapps.io/line/">реализация</a>  задания.

Визуализация:  
<p><img src="img\IZO.png" ></p>


## Plug-in алгоритм
 Нормальный дискриминантный анализ - это специальный случай байесовской классификации, предполагающий, что плотности всех классов являются многомерными нормальными.  
Случайная величина x имеет многомерное нормальное распределение, если ее плотность задается выражением:  


![](https://camo.githubusercontent.com/8e7cf0a285068cff21acc2a6d67cfaa81d85d184/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f4e253238782532432532302535436d752532432532302535435369676d612532392532302533442532302535436672616325374231253744253742253543737172742537422532383225354370692532392535456e2537432535435369676d61253743253744253744657870253238253543667261632537422d3125374425374232253744253238782532302d2532302535436d75253239253545542532302535435369676d612535452537422d31253744253238782532302d2532302535436d7525323925323925324325323078253230253543657073696c6f6e2532302535436d6174686262253742522537442535452537426e253744)  
где  𝜇 ∈ ℝ - математическое ожидание (центр), а  𝛴 ∈ ℝ - ковариационная матрица (симметричная, невырожденная, положительно определённая).  
Алгоритм заключается в том, чтобы найти неизвестные параметры 𝜇 и  𝛴  для каждого класса y и подставить их в формулу оптимального байесовского классификатора. В отличие от линейного дискриминанта Фишера(ЛДФ), в данном алгоритме мы предполагаем, что ковариационные матрицы не равны. 
Оценка параметров нормального распределения производится на основе параметров функций правдоподобия:  
![](https://camo.githubusercontent.com/94f89606a1238459d5fd563685797efb5709e35a/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f2535436d755f7925323025334425323025354366726163253742312537442537426c5f7925374425354373756d5f253742785f69253341795f6925323025334425323079253744253230785f69)  
![](https://camo.githubusercontent.com/cbc86c5d558986438255e75821f912a7df470d90/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f2535435369676d615f7925323025334425323025354366726163253742312537442537426c5f792532302d2532303125374425354373756d5f253742785f69253341795f6925323025334425323079253744253238785f692532302d2532302535436d755f79253239253238785f692532302d2532302535436d755f7925323925354554)  
Программная реализация восстановления данных параметров:  
Для математического ожидания 𝜇, то есть находим центр нормального распределения элементов класса:  

```diff  

  for (col in 1:cols){
   mu[1, col] = mean(objects[,col])
  }


```  
Для восстановления ковариационной матрицы 𝛴:  

```diff  

  for (i in 1:rows){
    sigma <- sigma + (t(objects[i,] - mu) %*% (objects[i,] - mu)) / (rows - 1)
  }


```     


Результаты работы подстановочного алгоритма:  
1. Здесь по 250 элементов в каждом классе и ковариационные матрицы равны, поэтому разделяющая линия, как видим, вырождается в прямую(данный случай рассматривается в алгоритме ЛДФ(Линейный дискриминант Фишера), который мы рассмотрим позже):    
![plugin2](https://user-images.githubusercontent.com/43229815/50246731-300d0880-03e7-11e9-83ef-8cc9b5bfc36d.png) 
2. Здесь 300 элементов в каждом классе. Видим, что эллипсоиды параллельны осям координат:  
Ковариационные матрицы:  
```diff  
Sigma1 <- matrix(c(10, 0, 0, 1), 2, 2)
Sigma2 <- matrix(c(3, 0, 0, 3), 2, 2)

```   
![plugin](https://user-images.githubusercontent.com/43229815/50246455-74e46f80-03e6-11e9-927d-4a930c0684a3.png)  
3. Здесь по 70 эелементов каждом классе. Ковариационные матрицы не равны, разделяющая плотность является квадратичной и прогибается таким образом, что менее плотный класс охватывает более плотный.  
Ковариационные матрицы:  
```diff  
Sigma1 <- matrix(c(3, 0, 0, 1), 2, 2)
Sigma2 <- matrix(c(10, 0, 0, 15), 2, 2)
``` 
![plugin1](https://user-images.githubusercontent.com/43229815/50246590-df95ab00-03e6-11e9-98de-0d1028f0c137.png)



## Линейный дискриминант Фишера
  
Алгоритм ЛДФ отличается от подстановочного алгоритма тем, что ковариационые матрицы классов равны, поэтому для их восстановления необходимо использовать все объекты выборки. В этом случае разделяющая кривая вырождается в прямую.  
Если оценить неизвестную 𝛴(ковариационная матрица, то есть их равенство), с учетом смещенности, то получим следующую формулу:  
![](https://camo.githubusercontent.com/a8d4c8e1eabfffb775b2e63c6e113c9e8e0f54ed/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f25354373756d2535452537422d2537442673706163653b3d2673706163653b25354366726163253742312537442537426c2d2537435925374325374425354373756d5f253742693d312537442535456c2673706163653b28785f692673706163653b2d2673706163653b2535436d752535452537422d2537445f253742795f692537442928785f692673706163653b2d2673706163653b2535436d752535452537422d2537445f253742795f692537442925354554)  
Восстановление ковариационных матриц в коде алгоритма:  
```diff  


    for (i in 1:rows1){
        sigma = sigma + (t(points1[i,] - mu1) %*% (points1[i,] - mu1))
	}

    for (i in 1:rows2){
        sigma = sigma + (t(points2[i,] - mu2) %*% (points2[i,] - mu2))
	}


```  
Разделяющая плоскость здается формулой:  
![](https://camo.githubusercontent.com/855779b58e3e1dcbdb989877d5fd3232da1d55f2/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543616c70686125323078253545542532302b2532302535436265746125323025334425323030),   
коэффициенты которой находятся следующим образом:  
![](https://camo.githubusercontent.com/863ea89add5cf8957b74c8efb86b0d69ae07c448/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f253543616c7068612532302533442532302535435369676d612535452537422d3125374425323025354363646f742532302532382535436d755f795f312532302d2532302535436d755f795f3225323925354554)  
![](https://camo.githubusercontent.com/65a258474daf14d2a0a47e3ae10e252e78cae132/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436265746125323025334425323025354366726163253742312537442537423225374425323025354363646f742532302535436d755f795f3125323025354363646f742532302535435369676d612535452537422d3125374425323025354363646f742532302535436d755f795f31253545542532302d25323025354366726163253742312537442537423225374425323025354363646f742532302535436d755f795f3225323025354363646f742532302535435369676d612535452537422d3125374425323025354363646f742532302535436d755f795f3225354554)  
Программная реализация данной функции нахождения коэффициентов ЛДФ выглядит следующим образом:  
```diff  
inverseSigma <- solve(Sigma)
alpha <- inverseSigma %*% t(mu1 - mu2)
beta <- (mu1 %*% inverseSigma %*% t(mu1) - mu2 %*% inverseSigma %*% t(mu2)) / 2
```  
  

Результат работы алгоритма выглядит следующим образом:  
![fisher1](https://user-images.githubusercontent.com/43229815/50239683-e9fa7980-03d3-11e9-9951-8c73bc48a399.png)  
Можно сравнить результаты работы алгоритмов с одинаковыми параметрами:

Здесь параметры следующие:  
```diff    
Sigma1 <- matrix(c(2, 0, 0, 2), 2, 2)
Sigma2 <- matrix(c(2, 0, 0, 2), 2, 2)  
```  
Количество элементов в каждом классе: 50.  
1.Подстановочный алгоритм.  
![pl](https://user-images.githubusercontent.com/43229815/50247232-64cd8f80-03e8-11e9-9c26-1206e4936ab5.png)  

2.ЛДФ алгоритм.  
![fisher2](https://user-images.githubusercontent.com/43229815/50240256-575ada00-03d5-11e9-9fb2-ce3a15dc0fbb.png)  
Видим, что превосходство ЛДФ очевидно. При чем, я заметила, что при малом количестве элементов в каждом классе ЛДФ превосходит подстановочный алгоритм. Чем меньше элементов, тем хуже работает подстановочный алгоритм.  


### Линейные алгоритмы классифиикации
======================================
Пусть ![](https://camo.githubusercontent.com/3612865dfb433fc39b8e49eb768f1afeced3cb41/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f582532302533442532302535436d6174686262253742522537442535452537426e253744253243253230592532302533442532302535436c6566742532302535432537422532302d312533422532302b312532302535437269676874253230253543253744), тогда алгоритм ![](https://camo.githubusercontent.com/d17f42554c08ca0cc4e040a9c288cb66ef4eec6d/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f6125323878253243253230772532392532302533442532307369676e2532302533437725324325323078253345) называется линейным алгоритмом.  
В данном пространстве классы разделяет гиперплоскость, которая задается уравнением:![](https://camo.githubusercontent.com/4760b2d7ccf75c8945f628d83440131cbf7b20a8/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436c616e676c65253230772532437825323025354372616e676c6525334430).  
Если x находится по одну сторону гиперплоскости с её направляющим вектором w, то объект x относится к классу +1, в противном случае - к классу -1.  


Эмпирический риск представлен следующей формулой: ![](https://camo.githubusercontent.com/212214890c628ad501d943664cdad24ffb1129df/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f6769662e6c617465783f5125323877253243253230582535456c25323925323025334425323025354373756d5f25374269253230253344253230312537442535452537426c2537444c25323825334377253243253230785f69253345795f69253239).  
Для того, чтобы минимизировать его и подобрать оптимальный вектор весов *w*, рекомендуется пользоваться методом стохастического градиента.  

**Метод стохастического градиента** - это итерационный процесс, на каждом шаге которого сдвигаемся в сторону, противоположную вектору
градиента 𝑄′(𝑤, 𝑋ℓ)) до тех пор, пока вектор весов 𝑤 не перестанет изменяться, причем вычисления градиента производится не на всех
объектах обучения, а выбирается случайный объект (отсюда и название метода «стохастический»), на основе которого и происходят
вычисления. В зависимости от функции потерь, которая используется в функционале эмпирического риска, будем получать различные
линейные алгоритмы классификации.  

Существует величина ![](https://camo.githubusercontent.com/9054359e62df82ec0a86de2ec203f14bbd3cdde8/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f4d5f6925323877253239253344795f692535436c616e676c65253230785f692532437725323025354372616e676c65), которая называется отступом объекта относительно алгоритма клссификации. Если данный отступ отрицательный, тогда алгоритм совершает ошибку.  

*L(M)* - функция потерь.  

Функции потерь для линейных алгоритмов классификации:  
**1.Адаптивный линейный элемент(ADALINE):**  
![](https://camo.githubusercontent.com/8972321c0046dede7d7689f0f75d795c710e90ea/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436d61746863616c2537424c2537442532384d2532392533442532384d2d31253239253545322533442532382535436c616e676c6525323077253243785f6925323025354372616e676c65253230795f692d3125323925354532) - это квадратичная функция потерь.  
![](https://camo.githubusercontent.com/aefe5920605d5b7b56b9ffe17f12f8816a79daae/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f77253344772d2535436574612532382535436c616e676c6525323077253243785f6925323025354372616e676c652d795f69253239785f69) - правило обновления весов на каждом шаге итерации метода стохастического градиента. Данное правило получено путем дифференцирования квадратичной функции.  
Программная реализация квадратичной функции потерь:  
```diff    
lossQuad <- function(x)
{
return ((x-1)^2)
} 
```  
**2.Персептрон Розенблатта:**  
![](https://camo.githubusercontent.com/97f2b6593c8f819f61676cb91aefc2a65784a8d4/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436d61746863616c2537424c2537442533442532382d4d2532395f2b2533442535436d61782532382d4d25324330253239) - данную функцию потерь называют кусочно-линейной.  
![](https://camo.githubusercontent.com/90d33699a1b3302dc1879c2c7eb823ff940b63f0/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437465787425374269662532302537442535436c616e676c6525323077253243785f6925323025354372616e676c65253230795f6925334330253230253543746578742537422532307468656e25323025374425323077253341253344772b253543657461253230785f69795f69) - правило обновления весов, которое называют правилом Хебба.  
Программная реализация функции потерь:  

```diff    
lossPerceptron <- function(x)
{
return (max(-x, 0))
}
```  
**3.Логистическая регрессия:**  
![](https://camo.githubusercontent.com/ee365cf43e497d5a900ef9c367bb83d742bdaecc/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535436d61746863616c2537424c2537442532384d2532392532302533442532302535436c6f675f32253238312532302b253230652535452537422d4d253744253239) - логистическая функция потерь.  
![](https://camo.githubusercontent.com/c1094cd0e0fefcaa0a2608da4bee189773cd1201/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f77253230253341253344253230772b253543657461253230795f69785f692535437369676d612532382d2535436c616e676c6525323077253243785f6925323025354372616e676c65253230795f69253239) - правило обновления весов, которое называют логистическим, где сигмоидная функция:  
![](https://camo.githubusercontent.com/a169c0ba965ef8fe5740bce9f2cd9d3ce47a5f38/687474703a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e6c617465783f2535437369676d612532387a2532392533442535436672616325374231253744253742312b652535452537422d7a253744253744).  
Программная реализация функции потерь и сигмоидной функции:  
```diff
##Функция потерь
lossLog <- function(x)
{
return (log2(1 + exp(-x)))
}
## Сигмоидная функция
sigmoidFunction <- function(z)
{
return (1 / (1 + exp(-z)))
}
```   
Адаптивный линейный элемент(ADALINE)  
--------------------------------  
Пусть дана обучающая выборка: множество входных значений X и множество выходящих значений Y, такие что каждому входу xj соответствует yj - выход, j = 1..m. Необходимо по этим данным построить ADALINE, которая допускает наименьшее количество ошибок на этой обучающей выборке. Обучение ADALINE заключается в подборе "наилучших" значений вектора весов w. Какие значение весов лучше определяет функционал потерь.  
В ADALINE применяется квадратичная функция потерь, которая представлена выше.  
Обучения ADALINE происходит следующим образом:  
Для начала нужно инициализировть веса * w_j; j = 0,..., n* и начальную оценку функционала *Q*.  
```diff
w <- c(1/2, 1/2, 1/2)
iterCount <- 0
## initialize Q
Q <- 0
for (i in 1:l)
{
## calculate the scalar product <w,x>
wx <- sum(w * xl[i, 1:n])
## calculate a margin
margin <- wx * xl[i, n + 1]
Q <- Q + lossQuad(margin)
}
```  
Выбрать объект *x_i* из *Xl*, затем посчитать ошибку и сделать шаг градиентного спуска:  
![](https://latex.codecogs.com/gif.latex?w%20%5C%2C%20%7B%3A%3D%7D%20w%20-%20%5Ceta%20%5C%28%20%28x_i%2C%20w%29%20-%20y_i%29*x_i)  

```diff
ex <- lossQuad(margin)
eta <- 1 / sqrt(sum(xi * xi))
w <- w - eta * (wx - yi) * xi ##шаг градиентного спуска
```  
Затем нужно оценить значение функционала *Q*:  
![](https://latex.codecogs.com/gif.latex?Q%20%5C%2C%20%7B%3A%3D%7D%20%5C%2C%20%281%20%5C%2C%20-%20%5C%2C%20%5Clambda%29Q%20%5C%2C%20&plus;%20%5C%2C%20%5Clambda%5Cvarepsilon_i)  

```diff
Qprev <- Q
Q <- (1 - lambda) * Q + lambda * ex
}
```   
Повторять все, что вычисляется после инициализации веса и начального значения *Q* пока значение *Q* не стабилизируется и/или веса *w* не перестанут изменяться.  
