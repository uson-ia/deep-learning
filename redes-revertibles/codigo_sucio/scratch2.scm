(import (scheme base)
	(srfi 27))

;;; ANN function
(define (sigmoid t)
  (/ 1 (+ 1 (exp (- t)))))

;;; ANN inverse function
(define (logit p)
  (- (log (- (/ 1 p) 1))))

;;; f and g are aliases for sigmoid and logit respectively
(define f sigmoid)
(define g logit)

;;; !! it is necessary that f^{-1} = g

;;; this is forward propagation for one neuron
(define (send-input f xs ws)
  (f (apply + (map * xs ws))))

;;; 
;;; sample neuron:
;;;

(define x0 1)
(define x1 .5)
(define x2 (/ 1 3.0))
(define x3 .5)

(define xs (list x0 x1 x2 x3))

(define w0 -1)
(define w1 4)
(define w2 -6)
(define w3 2)

(define ws (list w0 w1 w2 w3))

(define y (send-input f xs ws))

;;; if x1, x2 and x3 were unknowns and y was a given value
;;; the problem would change from:
;;;
;;; computing y
;;; 
;;; y = f(w0*x0 + w1*x1 + w2*x2 + w3*x3)
;;;
;;; to:
;;;
;;; solve for x1, x2 and x3 where 0 <= x1, x2, x3 <= 1 such that
;;;
;;;      g(y) - w0 - w2*x2 - w3*x3
;;; x1 = -------------------------
;;;                w1
;;;
;;;      g(y) - w0 - w1*x1 - w3*x3
;;; x2 = -------------------------
;;;                w2
;;;
;;;      g(y) - w0 - w1*x1 - w2*x2
;;; x3 = -------------------------
;;;                w3
;;; 
(define (send-output g y ws)
  (cons 1 (get-random-solution (g y) (car ws) (cdr ws))))

(define (get-random-solution s w0 wrest)
  (define (loop known-xs known-ws unknown-ws)
    (if (no-more? unknown-ws)
	'()
	(let* ((w (first unknown-ws))
	       (ks (map * known-xs known-ws))
	       (+s (map cut-neg unknown-ws))
	       (-s (map cut-pos unknown-ws))
	       (ba (/ (apply - `(,s ,w0 ,@ks ,@+s)) w))
	       (bb (/ (apply - `(,s ,w0 ,@ks ,@-s)) w))
	       (in (interval-cut ba bb))
	       (x (random-value in)))
	  (cons x (loop (cons x known-xs)
			(cons w known-ws)
			(rest unknown-ws))))))
  (loop '() '() wrest))

(define (unitary-intersect lo hi)
  (cons (if (> lo 0) lo 0)
	(if (< hi 1) hi 1)))

(define (random-value interval)
  (let ((lo (car interval))
	(hi (cdr interval)))
    (+ lo (* (random-real) (- hi lo)))))


;;;
;;; Example of inversion
;;;

;;; in the example s = g(y) = g(.5)
(define s (g y))

;;; bound A and B for x1
(define x1bound-a (/ (- s w0 (if (< w2 0) 0 w2) (if (< w3 0) 0 w3)) w1))
(define x1bound-b (/ (- s w0 (if (< w2 0) w2 0) (if (< w3 0) w3 0)) w1))

(define x1interval (unitary-intersect (min x1bound-a x1bound-b) (max x1bound-a x1bound-b)))

(define x1 (random-value x1interval))
; (define x1 0.5)

;;; bound A and B for x2
(define x2bound-a (/ (- s w0 (* w1 x1) (if (< w3 0) 0 w3)) w2))
(define x2bound-b (/ (- s w0 (* w1 x1) (if (< w3 0) w3 0)) w2))

(define x2interval (unitary-intersect (min x2bound-a x2bound-b) (max x2bound-a x2bound-b)))

(define x2 (random-value x2interval))
; (define x2 (/ 1 3.0))

;;; x3 is the last variable so we don't need bounds, we can just compute
(define x3 (/ (- s w0 (* w1 x1) (* w2 x2)) w3))

;;; -4e-15 ≈ 0
;;; -2e-15 ≈ 0
;;; 0      = 0
;;; 0      = 0
;;; -4e-15 ≈ 0
;;;
;;; These results are the x1 x2 and x3 values plugged in the equation

(define (no-more? xs)
  (null? xs))

(define (first xs)
  (car xs))

(define (rest xs)
  (cdr xs))

(define (cut-neg w)
  (if (< w 0) 0 w))

(define (cut-pos w)
  (if (< w 0) w 0))

(define (interval-cut a b)
  (unitary-intersect (min a b) (max a b)))

(define (magic s w0 known-xs known-ws unknown-ws)
  (if (no-more? unknown-ws)
      '()
      (let* (;(x  (first unknown-xs))
	     (w  (first unknown-ws))
	     (ks (apply + (map * known-xs known-ws)))
	     (+s (apply + (map cut-neg unknown-ws)))
	     (-s (apply + (map cut-pos unknown-ws)))
	     (ba (/ (- s w0 ks +s) w))
	     (bb (/ (- s w0 ks -s) w))
      	     (in (interval-cut ba bb))
	     (x  (random-value in)))
	(cons x (magic s
		       w0
		       (cons x known-xs)
		       (cons w known-ws)
		       ;(cdr unknown-xs)
		       (cdr unknown-ws))))))

(define (rand-error)
  (define xs (cons 1 (magic s w0 '() '() (cdr ws))))
  (abs (- y (send-input f xs ws))))
