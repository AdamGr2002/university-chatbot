import React from 'react';
import { SignIn } from '@clerk/clerk-react';

function SignInButton() {
  return (
    <SignIn>
      <SignIn.Button className="bg-indigo-600 text-white px-6 py-3 rounded hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition duration-150 ease-in-out">
        Sign In
      </SignIn.Button>
    </SignIn>
  );
}

export default SignInButton;
