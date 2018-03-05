# Basic learning testing
import numpy as np
import learn

def main():
    features = np.array([np.array([10*a for a in np.random.random(3)]) for _ in range(100)])
    labels = np.array([np.array([b*c for b, c in enumerate(a)]) for a in features])

    b = learn.Brain()
    b.train(features, labels)


    print "accuracy", b.evaluate(features, labels)
    print "expect", [1*0, 2*1, 3*2]
    print "got", b.predict(np.array([np.array([1.0,2.0,3.0])]))

if __name__ == '__main__':
    main()
