import { configureStore } from '@reduxjs/toolkit';
import measureReducer from './measures';
import appReducer from './app';
import parameterReducer from './parameters';
import seirReducer from './seir';

import thunk from 'redux-thunk';

export default configureStore({
  reducer: {
    measures: measureReducer,
    parameters: parameterReducer,
    seir: seirReducer,
    app: appReducer
  },
  middleware: [thunk]
});