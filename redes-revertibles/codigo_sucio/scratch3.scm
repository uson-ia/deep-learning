(import (scheme base)
	(srfi 27))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; NEURAL NETWORK PROCEDURES ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

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

;;; this is backward propagation for one neuron
(define (send-output g y ws)
  (define (get-random-solution s w0 wrest)
    (define (loop known-xs known-ws unknown-ws)
      (if (no-more? unknown-ws)
	  '()
	  (let* ((w  (first unknown-ws))
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
  (cons 1 (get-random-solution (g y) (car ws) (cdr ws))))


;;;;;;;;;;;;;;;;;;;;;;;;;;
;; AUXILIARY PROCEDURES ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (unitary-intersect lo hi)
  (cons (if (> lo 0) lo 0)
	(if (< hi 1) hi 1)))

(define (random-value interval)
  (let ((lo (car interval))
	(hi (cdr interval)))
    (+ lo (* (random-real) (- hi lo)))))

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

;;;;;;;;;;;;;
;; EXAMPLE ;;
;;;;;;;;;;;;;

(define x0 1.0)
(define x1 0.5)
(define x2 (/ 1 3.0))
(define x3 0.5)
(define xs (list x0 x1 x2 x3))

(define w0 -1.0)
(define w1 4.0)
(define w2 -6.0)
(define w3 2.0)
(define ws (list w0 w1 w2 w3))

(define y (send-input f xs ws))

(define (test-example)
  (let* ((y      (send-input f xs ws))
	 (new-xs (send-output g y ws))
	 (new-y  (send-input f new-xs ws)))
    (display "==================================\n")
    (display "TESTING ALGORITHMS\n")
    (display "==================================\n")
    (display "  ws = ")
    (display ws)
    (newline)
    (display "  original xs = ")
    (display xs)
    (newline)
    (display "  y = ")
    (display y)
    (newline)
    (display "  new xs = ")
    (display new-xs)
    (newline)
    (display "  new y = ")
    (display new-y)
    (newline)
    (display "---------------------------------\n")
    (display "error = ")
    (display (abs (- y new-y)))
    (newline)
    (display "==================================\n")))
