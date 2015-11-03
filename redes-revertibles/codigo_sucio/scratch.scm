(import (scheme base))

(import (srfi 27))

(define (logit p)
  (- (log (- (/ 1 p) 1))))

(define (sigmoid t)
  (/ 1 (+ 1 (exp (- t)))))

(define-record-type <neuron>
  (neuron f g)
  neuron?
  (f neuron-threshold)			; invertible unary operation
  (g neuron-threshold-inverse))		; inverse of f

(define (fold op id lst)
  (if (null? lst)
      id
      (op (car lst) (fold op id (cdr lst)))))

(define (send-input neuron inputs weights)
  (let ((f (neuron-threshold neuron)))
    (f (fold + 0 (map * inputs weights)))))

(define (random-weight)
  (- (* 200 (random-real)) 100))

(define (send-output neuron output weights intervals)
  'TODO)

(define (initial-intervals coefs)
  (if (null? coefs)
      '()
      (cons (cons 0 1) (initial-intervals (cdr coefs)))))


(define N (neuron sigmoid logit))


;;; finv(y) = w0 + w1*x1 + w2*x2 + w3*x3
;;;
;;; y = .5
;;; w0 = -1
;;; w1 = 4
;;; w2 = -6
;;; w3 = 2
;;;
;;; theta = finv(y) = 0.0
;;;
;;; i1 = [0 , 1]
;;; i2 = [0 , 1]
;;; i3 = [0 , 1]
;;;
;;; se computan las cotas para x1
;;;        theta - (w0) - (w2*x2) - (w3*x3)
;;; src =  --------------------------------
;;;                 (w1)
;;;
;;; se eligen los valores de x2 y x3 que maximicen (w2*x2) y (w3*x3) respectivamente
;;; para computar src-a
;;; 
;;; se eligen los valores de x2 y x3 que minimicen (w2*x2) y (w3*x3) respectivamente
;;; para computar src-b
;;;
;;; los intervalos de x1, x2 y x3 siempre serán a lo mas [0 , 1]; en caso que se desee
;;; maximizar el producto wi*xi donde el intervalo de xi es [xi-l , xi-u] se aborda
;;; por casos:
;;; * si wi es negativo para maximizar el producto se elige xi-l, si wi
;;;   es positivo para maximizar el producto se elige xi-u.
;;; * si wi es negativo para minimizar el producto se elige xi-u, si wi
;;;   es positivo para minimizar el producto se elige xi-l.

;; (define y .5)
;; (define weights (list -1 4 -6 2))
;; (define w0 (car weights))
;; (define ws (cdr weights))
;; (define theta (logit y))
;; (define is (initial-intervals ws))

(define (maximize-product w i)
  (if (negative? w)
      (car i)
      (cdr i)))

(define (minimize-product w i)
  (if (negative? w)
      (cdr i)
      (car i)))

;;; first iteration, compute the interval for x1
;;; ws-prev = ()
;;; is-prev = ()
;;; wi = w1
;;; ii = i1
;;; ws-next = (w2 w3)
;;; is-next = (i2 i3)
;; (define ws-prev '())
;; (define is-prev '())
;; (define wi (car ws))
;; (define ii (car is))
;; (define ws-next (cdr ws))
;; (define is-next (cdr is))
;; (define max-prods (map maximize-product ws-next is-next))
;; (define min-prods (map minimize-product ws-next is-next))

;; (define a (/ (apply - `(,theta ,w0 ,@(map * max-prods ws-next))) wi))
;; (define b (/ (apply - `(,theta ,w0 ,@(map * min-prods ws-next))) wi))

(define (decide-interval a b i)
  (define lo (car i))
  (define hi (cdr i))
  (define nlo (min a b))
  (define nhi (max a b))
  (if (or (> (min a b) hi)
	  (< (max a b) lo))
      (error "The problem doesn't have a solution")
      (cons (if (< nlo lo) lo nlo)
	    (if (> nhi hi) hi nhi))))


;;; next iteration, compute the interval for x2
;;; ws-prev = (w1)
;;; is-prev = (i1)
;;; wi = w2
;;; ii = i2
;;; ws-next = (w3)
;;; is-next = (i3)

(define (compute-intervals theta w0 ws-prev is-prev wi ii ws-next is-next)
  (if (null? wi)
      (reverse is-prev)
      (let ((max-prods-prev (map maximize-product ws-prev is-prev))
	    (min-prods-prev (map minimize-product ws-prev is-prev))
	    (max-prods-next (map maximize-product ws-next is-next))
	    (min-prods-next (map minimize-product ws-next is-next)))
	(let ((a (/ (apply - `(,theta ,w0 ,@(map * max-prods-prev ws-prev) ,@(map * max-prods-next ws-next))) wi))
	      (b (/ (apply - `(,theta ,w0 ,@(map * min-prods-prev ws-prev) ,@(map * min-prods-next ws-next))) wi)))
	  (compute-intervals theta
			     w0
			     (cons wi ws-prev)
			     (cons (decide-interval a b ii) is-prev)
			     (if (null? ws-next) '() (car ws-next))
			     (if (null? ws-next) '() (car is-next))
			     (if (null? ws-next) '() (cdr ws-next))
			     (if (null? ws-next) '() (cdr is-next)))))))

;;; quiero tener una función que dependa de un valor de entrada <y> y calcule una solución aleatoria dados los
;;; pesos ws (tamaño n+1)

(define (random-value interval)
  (let ((lo (car interval))
	(hi (cdr interval)))
    (+ lo (* (random-real) (- hi lo)))))

(define (magic y ws)
  (define w0 (car ws))
  (define theta (logit y))
  (define w1 (cadr ws))
  (define i1 (cons 0 1))
  (define ws (cddr ws))
  (define is (initial-intervals ws))
  (define new-is (compute-intervals theta w0 '() '() w1 i1 ws is))
  (define x2..n (map random-value (cdr new-is)))
  (define x1 (/ (apply - `(,theta ,w0 ,@x2..n)) w1))
  (cons x1 x2..n))

;;; y = .5
;;; w0 = -1
;;; w1 = 4
;;; w2 = -6
;;; w3 = 2
(define y 0.5)
(define theta (logit y))
(define w0 -1)
(define w1 4)
(define w2 -6)
(define w3 2)


(define ws (list w1 w2 w3))
(define is (initial-intervals ws))


(define new-is (compute-intervals theta w0 '() '() (car ws) (car is) (cdr ws) (cdr is)))

;; finv(y) = w0 + w1*x1 + w2*x2 + w3*x3

(define (try theta w0 ws)
  (lambda xs
    (= theta (+ w0 (apply + (map * ws xs))))))


(define sample-try (try theta w0 ws))
