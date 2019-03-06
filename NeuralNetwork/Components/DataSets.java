package NeuralNetwork.Components;

import java.util.ArrayList;
import java.util.Iterator;

/**
 * DataSets -
 * The DataSets maintains all DataSet(s)
 * and is capable of returning a single
 * DataSet or a sub set of DataSet(s).
 *
 * @author Luca Jung
 * @version 2.1
 */
public class DataSets implements Iterable<DataSet> {

    private ArrayList<DataSet> dataSets;

    public DataSets(){
        dataSets = new ArrayList<>();
    }

    public void addDataSet(DataSet dataSet){
        dataSets.add(dataSet);
    }

    public DataSet getDataSet(int index){
        if(index >= 0 && index < size()) {
            return dataSets.get(index);
        }
        else {
            throw new IndexOutOfBoundsException();
        }
    }

    public DataSets getRandomSubSet(int subSetSize){
        if(subSetSize > 0 && subSetSize <= size()) {
            DataSets ds = new DataSets();
            for (int i = 0; i < subSetSize; i++) {
                int randomInt = MathTools.getRandomInt(0, size() - 1);
                DataSet dataSet = getDataSet(randomInt);
                ds.addDataSet(dataSet);
            }
            return ds;
        }
        else {
            throw new IllegalArgumentException();
        }
    }

    public int size(){
        return dataSets.size();
    }

    @Override
    public Iterator<DataSet> iterator() {
        return new DataSetsIterator();
    }

    public class DataSetsIterator implements Iterator {

        private Iterator iterator;

        public DataSetsIterator(){
            iterator = dataSets.iterator();
        }

        @Override
        public Object next() {
            return iterator.next();
        }

        @Override
        public boolean hasNext() {
            return iterator.hasNext();
        }
    }
}
